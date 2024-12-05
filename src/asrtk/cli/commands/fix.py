"""Command for fixing punctuation and grammar in VTT files using Claude AI."""
from pathlib import Path
import rich_click as click
import os
import json
from datetime import datetime
from typing import Dict, Any
from anthropic import Anthropic

from ...core.vtt import split_vtt_into_chunks, combine_vtt_chunks
from ...utils.file import backup_file, load_cache, save_cache

def create_prompt(content: str, lang: str = "Turkish") -> str:
    """Create the prompt for Claude.

    Args:
        content: VTT content to fix
        lang: Language of the content

    Returns:
        Formatted prompt for Claude
    """
    return f"""Please fix punctuation and grammar in this {lang} VTT subtitle file. Important rules:

1. Preserve all timestamp lines (HH:MM:SS.mmm --> HH:MM:SS.mmm) exactly as they are
2. Do not add periods when a sentence continues in the next subtitle segment
3. Only add periods when a complete sentence ends
4. Keep the subtitle structure intact - each segment must stay with its timestamp
5. Fix any spelling mistakes and grammar issues
6. Maintain line breaks within subtitle segments
7. Do not add explanations or comments

Example of continuing sentences:
WEBVTT

00:00:01.000 --> 00:00:03.000
Bugün size anlatacağım konu

00:00:03.100 --> 00:00:05.000
çok önemli bir mesele.

Here's the subtitle content to fix:

{content}"""

@click.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.option(
    "--model",
    type=click.Choice([
        "claude-3-haiku-20240307",
        "claude-3-sonnet-20240229",
        "claude-3.5-sonnet-20240229"
    ]),
    default="claude-3-haiku-20240307",
    help="Anthropic model to use"
)
@click.option("--max-tokens", default=3500, type=int, help="Maximum tokens per API call")
@click.option("--chunk-size", default=2000, type=int, help="Chunk size in tokens (will be adjusted for prompt)")
@click.option("--cache-file", type=str, default="fix_cache.json", help="Cache filename")
def fix(input_dir: str, model: str, max_tokens: int, chunk_size: int, cache_file: str) -> None:
    """Fix punctuation and grammar in VTT subtitle files using Claude AI."""
    start_time = datetime.now()

    # Initialize cache in the input directory
    input_dir = Path(input_dir)
    cache_path = input_dir / cache_file
    cache = load_cache(cache_path)
    chunks_processed = 0
    chunks_from_cache = 0
    api_calls = 0

    # Model token limits and pricing info
    MODEL_LIMITS = {
        "claude-3-haiku-20240307": 4096,
        "claude-3-sonnet-20240229": 200000,
        "claude-3.5-sonnet-20240229": 200000
    }

    MODEL_PRICING = {
        "claude-3-haiku-20240307": "($0.25/M input, $1.25/M output)",
        "claude-3-sonnet-20240229": "($3/M input, $15/M output)",
        "claude-3.5-sonnet-20240229": "($3/M input, $15/M output)"
    }

    click.echo(f"Using model: {model} {MODEL_PRICING[model]}")

    # Reserve tokens for prompt and some padding
    PROMPT_TOKENS = 150  # Approximate tokens for the prompt
    PADDING_TOKENS = 100  # Safety padding

    # Adjust chunk size based on model
    if model == "claude-3-haiku-20240307":
        effective_chunk_size = min(
            chunk_size,
            MODEL_LIMITS[model] - PROMPT_TOKENS - PADDING_TOKENS
        )
    else:
        # For Sonnet models, use a larger chunk size
        effective_chunk_size = min(
            50000,  # Reasonable default for large models
            MODEL_LIMITS[model] - PROMPT_TOKENS - PADDING_TOKENS
        )

    if max_tokens > MODEL_LIMITS[model]:
        click.echo(f"Warning: Reducing max_tokens to {MODEL_LIMITS[model]} for {model}")
        max_tokens = MODEL_LIMITS[model]

    # Load API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("ANTHROPIC_API_KEY")
        except ImportError:
            click.echo("python-dotenv not installed. Using environment variables only.")

    if not api_key:
        raise click.UsageError("ANTHROPIC_API_KEY not found in environment variables or .env file")

    client = Anthropic(api_key=api_key)
    input_dir = Path(input_dir)
    vtt_files = list(input_dir.glob("*.vtt"))

    if not vtt_files:
        click.echo("No VTT files found in the input directory")
        return

    for vtt_file in vtt_files:
        click.echo(f"\nProcessing {vtt_file.name}...")
        content = vtt_file.read_text(encoding='utf-8')

        estimated_tokens = len(content.split()) * 1.3
        click.echo(f"Estimated tokens: {int(estimated_tokens)}")

        # Only chunk if necessary based on model and content size
        needs_chunking = estimated_tokens + PROMPT_TOKENS + PADDING_TOKENS > MODEL_LIMITS[model]

        if needs_chunking:
            chunks = split_vtt_into_chunks(content, effective_chunk_size)
            total_chunks = len(chunks)
            click.echo(f"Content too large for {model}, split into {total_chunks} chunks")
        else:
            chunks = [content]
            total_chunks = 1
            click.echo(f"Processing entire file in one call ({int(estimated_tokens)} tokens)")

        fixed_chunks = []
        for i, chunk in enumerate(chunks, 1):
            chunk_hash = get_chunk_hash(chunk)
            chunks_processed += 1

            # Try to get from cache
            if chunk_hash in cache:
                fixed_chunks.append(cache[chunk_hash])
                chunks_from_cache += 1
                click.echo(f"Chunk {i}/{total_chunks}: Using cached version ✓")
                continue

            if total_chunks > 1:
                click.echo(f"Chunk {i}/{total_chunks}: Processing with API...")
            else:
                click.echo("Processing with API...")

            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{
                        "role": "user",
                        "content": create_prompt(chunk, lang="Turkish")
                    }]
                )
                fixed_content = response.content[0].text
                api_calls += 1

                # Cache the result
                cache[chunk_hash] = fixed_content
                save_cache(cache, cache_path)

                fixed_chunks.append(fixed_content)
                click.echo(f"API call successful ✓")

            except Exception as e:
                click.echo(f"Error processing: {e}", err=True)
                if "maximum allowed number of tokens" in str(e):
                    if model != "claude-3-haiku-20240307":
                        click.echo("Consider using a smaller chunk size or switching to Haiku model")
                    # ... rest of error handling remains the same ...

        fixed_content = combine_vtt_chunks(fixed_chunks)

        # Backup and save
        backup_path = backup_file(str(vtt_file))
        vtt_file.write_text(fixed_content, encoding='utf-8')
        click.echo(f"✓ Fixed and saved: {vtt_file.name}")
        click.echo(f"  Backup created: {Path(backup_path).name}")

    # Print summary
    duration = datetime.now() - start_time
    click.echo("\nSummary:")
    click.echo(f"Total chunks processed: {chunks_processed}")
    click.echo(f"Chunks from cache: {chunks_from_cache}")
    click.echo(f"API calls made: {api_calls}")
    click.echo(f"Total time: {duration}")
    click.echo(f"Cache file: {cache_path}")

def get_chunk_hash(chunk: str) -> str:
    """Generate a hash for the chunk content.

    Args:
        chunk: Content to hash

    Returns:
        SHA-256 hash of the content
    """
    import hashlib
    return hashlib.sha256(chunk.encode()).hexdigest()
