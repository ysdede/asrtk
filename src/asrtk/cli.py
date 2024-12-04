"""Main CLI for asrtk."""
import rich_click as click
import json
import re
from pathlib import Path
import yt_dlp
from .utils import is_valid_json_file, backup_file
from .downloader import vid_info, check_sub_lang, ydl_opts, check_audio_lang, check_video_langs

from . import __version__

import os
from anthropic import Anthropic
from .utils import backup_file

import hashlib
from datetime import datetime
from typing import Optional, Dict
from collections import Counter
import csv
from rich import print as rprint
from rich.console import Console
from rich.text import Text

context_settings = {"help_option_names": ["-h", "--help"]}
help_config = click.RichHelpConfiguration(
    width=88,
    show_arguments=True,
    use_rich_markup=True,
)


@click.group(context_settings=context_settings)
@click.rich_config(help_config=help_config)
@click.version_option(__version__, "-v", "--version")
def cli() -> None:
    """An open-source Python toolkit designed to streamline the development and enhancement of ASR systems."""


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path(exists=False))
@click.option("--audio_type", default="m4a", help="Audio file type, default is 'm4a'.")
@click.option("--pt", default=10, type=int, help="Optional parameter, default is 10.")
@click.option("--tolerance", default=250.0, type=float, help="Tolerance value, default is 250.0.")
# @click.option("--force-merge", default=False, type=bool, help="Force merge, default is False.")
@click.option("--forced-alignment", default=True, type=bool, help="Force alignment, default is True.")
@click.option("-fm", "--force-merge", is_flag=True, help="Force merge, default is False.")
def split(input_dir, output_dir, audio_type, pt, tolerance, forced_alignment, force_merge: bool=False):
    """ Command to split audio files. """
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Audio type: {audio_type}")
    print(f"Optional parameter pt: {pt}")
    print(f"Tolerance: {tolerance}")
    print(f"Forced alignment: {forced_alignment}")
    print(f"Forced merge: {force_merge}")

    from pathlib import Path
    from asrtk.subs import split_audio_with_subtitles
    from asrtk.utils import natural_sort_key

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    inner_folder_name = input_dir.name

    # Check if the output_dir does not already end with the inner folder name
    if not output_dir.name == inner_folder_name:
        # Append the inner folder name to the output_dir
        output_dir = output_dir / inner_folder_name

    output_dir.mkdir(parents=True, exist_ok=True)

    audio_file_filter = f'*.{audio_type}'
    # Collect all .vtt and .m4a files in the directory
    vtt_files = [file.name for file in input_dir.glob("*.vtt")]
    acc_files = [file.name for file in input_dir.glob(audio_file_filter)]

    acc_files = sorted(acc_files, key=natural_sort_key)

    pairs = []
    for acc_file in acc_files:
        for vtt_file in vtt_files:
            if acc_file.split(".")[0] == vtt_file.split(".")[0]:
                pairs.append((acc_file, vtt_file))

    print(f"{output_dir}")

    for p in pairs:
        # split filename with ., drop last element, recompile as string back
        episode_name = ".".join(p[0].split(".")[:-1])
        print(p, episode_name)

        episode_dir = output_dir / episode_name
        episode_dir.mkdir(exist_ok=True)
        # print(f'{input_dir}/{p[1]}',f'{input_dir}/{p[0]}')
        print(episode_dir)
        split_audio_with_subtitles(f"{input_dir}/{p[1]}", f"{input_dir}/{p[0]}", episode_dir, tolerance=tolerance, period_threshold=pt, force_merge=force_merge, forced_alignment=forced_alignment)


@cli.command()
@click.argument("input_", metavar="INPUT")
@click.option(
    "-r",
    "--reverse",
    is_flag=True,
    help="Reverse the input.",
)
def repeat(input_: str, *, reverse: bool = False) -> None:
    """Repeat the input."""
    click.echo(input_ if not reverse else input_[::-1])


@cli.command()
@click.argument("work_dir", type=click.Path(exists=True))
@click.option("-u", "--url", "playlist_url", type=str, help="YouTube playlist URL")
@click.option("-f", "--file", "playlist_file", type=click.Path(exists=True), help="File containing playlist URLs")
def download_playlist(work_dir: str, playlist_url: str | None, playlist_file: str | None) -> None:
    """Download videos from a YouTube playlist with subtitles.

    Provide either a playlist URL using --url or a file containing URLs using --file
    """
    if not playlist_file and not playlist_url:
        raise click.UsageError("Either --url or --file must be specified.")
    if playlist_url and playlist_file:
        raise click.UsageError("Cannot specify both --url and --file. Choose one.")

    work_dir = Path(work_dir)
    json_info_dir = work_dir / "json_info"
    json_info_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts['paths']['home'] = str(work_dir)

    # Determine the source of playlist URLs
    if playlist_url:
        playlist_urls = [playlist_url]
    else:
        playlist_urls = [line.strip() for line in Path(playlist_file).read_text().splitlines()]

    total_videos = 0
    downloaded = 0

    for url in playlist_urls:
        if not url:
            continue  # Skip empty lines in the playlist file

        playlist_id = url.split('list=')[-1].split('&')[0]
        json_file = json_info_dir / f"{playlist_id}.json"

        if json_file.exists() and is_valid_json_file(json_file):
            click.echo(f"Loading existing playlist info from {json_file}")
            pl_info = json.loads(json_file.read_text())
        else:
            pl_info = vid_info(url)
            click.echo(f"Processing {pl_info['title']}, {pl_info['id']}, episodes: {len(pl_info['entries'])}...")
            json_file.write_text(json.dumps(pl_info, indent=4))

        click.echo(f"{pl_info['title']}, {pl_info['id']}")

        playlist_dir = work_dir / re.sub(r'[^a-zA-Z0-9]', '_', pl_info["title"])
        playlist_dir.mkdir(exist_ok=True)
        ydl_opts['paths']['home'] = str(playlist_dir)
        ydl_opts['download_archive'] = str(playlist_dir / 'downloaded.txt')

        for entry in pl_info['entries']:
            if not entry:  # Skip empty/private videos
                continue

            total_videos += 1
            video_id = entry.get('id') or entry.get('url', '').split('watch?v=')[-1]
            if not video_id:
                click.echo(f"Skipping entry with no video ID: {entry}")
                continue

            try:
                with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                    video_info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)

                has_subs, has_audio = check_video_langs(info=video_info)
                if has_subs and has_audio:
                    click.echo(f"Processing: {entry.get('title', video_id)}, ✔ (TR audio + subs)")
                    try:
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
                        downloaded += 1
                    except Exception as e:
                        click.echo(f"Error downloading {entry.get('title', video_id)}: {e}", err=True)
                else:
                    status = []
                    if not has_subs: status.append("no TR subs")
                    if not has_audio: status.append("no TR audio")
                    click.echo(f"Skipping: {entry.get('title', video_id)}, ❌ ({', '.join(status)})")

            except Exception as e:
                click.echo(f"Error checking video {video_id}: {e}", err=True)
                continue

    click.echo(f"\nDownload complete. {downloaded}/{total_videos} videos processed.")


@cli.command()
@click.argument("work_dir", type=click.Path(exists=True))
@click.option("-u", "--url", "channel_url", type=str, help="YouTube channel URL")
@click.option("-f", "--file", "channel_file", type=click.Path(exists=True), help="File containing channel info JSON")
def download_channel(work_dir: str, channel_url: str | None, channel_file: str | None) -> None:
    """Download videos from a YouTube channel with subtitles.

    Provide either a channel URL using --url or a pre-downloaded channel info file using --file
    """
    if not channel_file and not channel_url:
        raise click.UsageError("Either --url or --file must be specified.")
    if channel_url and channel_file:
        raise click.UsageError("Cannot specify both --url and --file. Choose one.")

    work_dir = Path(work_dir)
    json_info_dir = work_dir / "json_info"
    json_info_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts['paths']['home'] = str(work_dir)

    # Get channel info either from URL or file
    if channel_url:
        # Extract and sanitize channel ID/name
        if '/c/' in channel_url:
            channel_id = channel_url.split('/c/')[-1]
        elif '/@' in channel_url:
            channel_id = channel_url.split('/@')[-1]
        elif '/channel/' in channel_url:
            channel_id = channel_url.split('/channel/')[-1]
        else:
            channel_id = channel_url.split('/')[-1]

        # Remove URL parameters if any
        channel_id = channel_id.split('?')[0]
        # Sanitize the channel ID for filename
        channel_id = re.sub(r'[^a-zA-Z0-9_-]', '_', channel_id)
        json_file = json_info_dir / f"{channel_id}.json"

        if json_file.exists() and is_valid_json_file(json_file):
            click.echo(f"Loading existing channel info from {json_file}")
            channel_info = json.loads(json_file.read_text())
        else:
            # Configure yt-dlp options for channel extraction
            channel_opts = {
                'quiet': True,
                'extract_flat': True,
                'ignoreerrors': True,
                'dump_single_json': True,
                'playlistreverse': False,
                'playlistend': None,
                'simulate': True,
                'skip_download': True,
                'no_warnings': True,
                'extract_flat': 'in_playlist',
                'playlist_items': '1-1000',  # Adjust if needed for larger channels
            }

            # First get channel info and videos list
            with yt_dlp.YoutubeDL(channel_opts) as ydl:
                # Add /videos to channel URL to get all uploads
                channel_vids_url = f"{channel_url}/videos"
                channel_info = ydl.extract_info(channel_vids_url, download=False)

            click.echo(f"Saving channel info to {json_file}")
            json_file.write_text(json.dumps(channel_info, indent=4))
    else:
        channel_info = json.loads(Path(channel_file).read_text())

    # Create channel directory
    channel_dir = work_dir / re.sub(r'[^a-zA-Z0-9]', '_', channel_info["title"])
    channel_dir.mkdir(exist_ok=True)
    ydl_opts['paths']['home'] = str(channel_dir)
    ydl_opts['download_archive'] = str(channel_dir / 'downloaded.txt')

    # Process all entries/videos in the channel
    total_videos = 0
    downloaded = 0

    # Get all video entries
    entries = channel_info.get('entries', [])

    click.echo(f"Found {len(entries)} videos in channel {channel_info['title']}")

    for entry in entries:
        if not entry:  # Skip empty/private videos
            continue

        total_videos += 1
        video_id = entry.get('id') or entry.get('url', '').split('watch?v=')[-1]
        if not video_id:
            click.echo(f"Skipping entry with no video ID: {entry}")
            continue

        # Get full video info to check subtitles
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                video_info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)

            has_subs, has_audio = check_video_langs(info=video_info)

            if has_subs and has_audio:
                click.echo(f"Processing: {entry.get('title', video_id)}, ✔ (TR audio + subs)")
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
                    downloaded += 1
                except Exception as e:
                    click.echo(f"Error downloading {entry.get('title', video_id)}: {e}", err=True)
            else:
                status = []
                if not has_subs: status.append("no TR subs")
                if not has_audio: status.append("no TR audio")
                click.echo(f"Skipping: {entry.get('title', video_id)}, ❌ ({', '.join(status)})")

        except Exception as e:
            click.echo(f"Error checking video {video_id}: {e}", err=True)
            continue

    click.echo(f"\nDownload complete. {downloaded}/{total_videos} videos processed.")


def get_chunk_hash(chunk: str) -> str:
    """Generate a hash for the chunk content."""
    return hashlib.sha256(chunk.encode()).hexdigest()

def load_cache(cache_file: Path) -> Dict:
    """Load the cache from file if it exists."""
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except json.JSONDecodeError:
            click.echo("Warning: Cache file corrupted, starting fresh")
    return {}

def save_cache(cache: Dict, cache_file: Path) -> None:
    """Save the cache to file."""
    cache_file.write_text(json.dumps(cache, indent=2))

@cli.command()
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
def fix(input_dir: str, model: str, max_tokens: int, chunk_size: int, cache_file: str):
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

def create_prompt(content: str, lang: str = "Turkish") -> str:
    """Create the prompt for Claude."""
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

def split_vtt_into_chunks(content: str, chunk_size: int) -> list[str]:
    """Split VTT content into chunks while preserving VTT structure and subtitle blocks."""
    lines = content.split('\n')
    chunks = []
    current_chunk = []
    current_size = 0
    subtitle_block = []
    in_subtitle = False

    # Always include VTT header in each chunk
    header = lines[0] if lines and lines[0].strip().upper() == 'WEBVTT' else 'WEBVTT\n'

    for line in lines[1:]:  # Skip WEBVTT header
        # Start of a subtitle block (timestamp line)
        if '-->' in line:
            in_subtitle = True
            if subtitle_block:
                subtitle_block.append('')  # Add blank line between subtitles
            subtitle_block = [line]
            continue

        if in_subtitle:
            if not line.strip():  # End of subtitle block
                in_subtitle = False
                block_size = sum(len(l.split()) for l in subtitle_block)

                # Check if adding this block would exceed chunk size
                if current_size + block_size > chunk_size and current_chunk:
                    # Complete current chunk
                    chunks.append('\n'.join([header] + current_chunk))
                    current_chunk = []
                    current_size = 0

                current_chunk.extend(subtitle_block)
                current_size += block_size
                subtitle_block = []
            else:
                subtitle_block.append(line)

    # Add any remaining subtitle block
    if subtitle_block:
        current_chunk.extend(subtitle_block)

    # Add any remaining content
    if current_chunk:
        chunks.append('\n'.join([header] + current_chunk))

    return chunks or [content]  # Return original content if no chunks were created

def combine_vtt_chunks(chunks: list[str]) -> str:
    """Combine VTT chunks while handling overlapping timestamps."""
    # Remove WEBVTT header from all but first chunk
    result = chunks[0]
    for chunk in chunks[1:]:
        # Remove header and any leading blank lines
        lines = chunk.split('\n')
        while lines and (not lines[0].strip() or lines[0].strip().upper() == 'WEBVTT'):
            lines.pop(0)
        result += '\n' + '\n'.join(lines)
    return result

@cli.command()
@click.argument("work_dir", type=click.Path(exists=True))
@click.option("--output", "-o", type=str, default="charset.json", help="Output JSON file name")
@click.option("--extra-sequences", "-e", multiple=True, help="Additional character sequences to include")
@click.option("--rare-threshold", "-r", type=int, default=2,
              help="Frequency threshold for rare characters (default: 2)")
@click.option("--create-delete-batch", "-d", is_flag=True,
              help="Create batch file for deleting files with rare characters")
def create_charset(work_dir: str, output: str, extra_sequences: tuple[str, ...],
                  rare_threshold: int, create_delete_batch: bool) -> None:
    """Create character set statistics from VTT files.

    Analyzes all VTT files recursively in the work directory and creates a JSON file with:
    - Character frequencies
    - File locations for rare characters
    - Additional character sequences specified with -e option

    Can optionally create a batch file to delete files containing rare characters
    (characters appearing less than the specified threshold).

    Example:
        asrtk create-charset ./vtt_files -e " ?" -e " !" -r 3 -d
    """
    work_dir = Path(work_dir)

    # Find all VTT files recursively
    vtt_files = list(work_dir.rglob("*.vtt"))

    if not vtt_files:
        click.echo("No VTT files found in the work directory")
        return

    click.echo(f"Found {len(vtt_files)} VTT files")

    # Process all VTT files and count character frequencies
    char_counts: Dict[str, int] = Counter()
    sequence_counts: Dict[str, int] = Counter()
    char_locations: Dict[str, Dict[str, int]] = {}  # char -> {filename -> count}
    files_to_delete = set()  # Track files containing rare chars
    total_chars = 0

    with click.progressbar(vtt_files, label='Processing VTT files') as files:
        for vtt_file in files:
            try:
                content = vtt_file.read_text(encoding='utf-8')
                file_chars = Counter(content)

                # Update global counts
                char_counts.update(file_chars)
                total_chars += len(content)

                # Track file locations for each character
                for char, count in file_chars.items():
                    if char not in char_locations:
                        char_locations[char] = {}
                    char_locations[char][str(vtt_file)] = count

                # Count extra sequences
                for seq in extra_sequences:
                    count = content.count(seq)
                    if count > 0:
                        sequence_counts[seq] += count
                        if seq not in char_locations:
                            char_locations[seq] = {}
                        char_locations[seq][str(vtt_file)] = count

            except Exception as e:
                click.echo(f"\nError processing {vtt_file}: {e}", err=True)
                continue

    # Find files containing rare characters
    if create_delete_batch:
        for char, count in char_counts.items():
            if count < rare_threshold:
                # Add all files containing this rare char to deletion set
                files_to_delete.update(char_locations[char].keys())

    # Create batch file if requested
    if create_delete_batch and files_to_delete:
        batch_file = Path(output).with_suffix('.bat')
        files_list = Path(output).with_suffix('.txt')

        # Save list of files
        with files_list.open('w', encoding='utf-8') as f:
            for file in sorted(files_to_delete):
                f.write(f"{file}\n")

        # Create batch file
        with batch_file.open('w', encoding='utf-8') as f:
            f.write("@echo off\n")
            f.write(f"echo Deleting {len(files_to_delete)} files containing rare characters...\n")
            for file in sorted(files_to_delete):
                f.write(f'del "{file}"\n')
            f.write("echo Done.\n")
            f.write("pause\n")

        click.echo(f"\nFound {len(files_to_delete)} files containing rare characters")
        click.echo(f"File list saved to: {files_list}")
        click.echo(f"Batch file saved to: {batch_file}")

    # Prepare JSON structure
    charset_data = {
        "stats": {
            "total_chars": total_chars,
            "unique_chars": len(char_counts),
            "extra_sequences": len(sequence_counts),
            "rare_threshold": rare_threshold if create_delete_batch else None,
            "files_with_rare_chars": len(files_to_delete) if create_delete_batch else None
        },
        "characters": [],
        "sequences": []
    }

    # Add single characters
    for char, count in sorted(char_counts.items(), key=lambda x: x[1], reverse=True):
        char_info = {
            "char": char,
            "frequency": count,
            "replacement": "",
            "percentage": (count / total_chars) * 100
        }

        # Add file locations for rare characters (count <= 5)
        if count <= 5:
            char_info["locations"] = {
                str(k): v for k, v in char_locations[char].items()
            }

            # For rare chars, add sample sentences from the files
            samples = []
            for file_path in char_locations[char]:
                try:
                    content = Path(file_path).read_text(encoding='utf-8')
                    file_samples = find_sample_sentences(content, char)
                    for sample in file_samples:
                        if sample not in samples:  # Avoid duplicates
                            samples.append(sample)
                            if len(samples) >= 3:  # Limit to 3 samples total
                                break
                    if len(samples) >= 3:
                        break
                except Exception as e:
                    click.echo(f"\nError reading {file_path}: {e}", err=True)
                    continue

            if samples:
                char_info["samples"] = samples

        # For frequent chars (count > 5), take samples from a random selection of files
        else:
            # Get up to 3 random files containing this char
            file_paths = list(char_locations[char].keys())
            if len(file_paths) > 3:
                from random import sample
                file_paths = sample(file_paths, 3)

            samples = []
            for file_path in file_paths:
                try:
                    content = Path(file_path).read_text(encoding='utf-8')
                    file_samples = find_sample_sentences(content, char)
                    for sample in file_samples:
                        if sample not in samples:  # Avoid duplicates
                            samples.append(sample)
                            if len(samples) >= 3:  # Limit to 3 samples total
                                break
                    if len(samples) >= 3:
                        break
                except Exception as e:
                    continue

            if samples:
                char_info["samples"] = samples

        charset_data["characters"].append(char_info)

    # Add sequences
    for seq, count in sorted(sequence_counts.items(), key=lambda x: x[1], reverse=True):
        seq_info = {
            "sequence": seq,
            "frequency": count,
            "replacement": "",
            "percentage": (count / total_chars) * 100,
            "locations": {
                str(k): v for k, v in char_locations[seq].items()
            }
        }
        charset_data["sequences"].append(seq_info)

    # Save to JSON file
    output_file = Path(output)
    with output_file.open('w', encoding='utf-8') as f:
        json.dump(charset_data, f, ensure_ascii=False, indent=2)

    click.echo(f"\nProcessed {total_chars:,} total characters")
    click.echo(f"Found {len(char_counts):,} unique characters")
    click.echo(f"Added {len(sequence_counts)} extra sequences")
    click.echo(f"Results saved to {output_file}")

    # Display top 10 most frequent characters/sequences
    click.echo("\nTop 10 most frequent characters:")
    sorted_items = (
        [(char, count) for char, count in char_counts.items()] +
        [(seq, count) for seq, count in sequence_counts.items()]
    )
    sorted_items.sort(key=lambda x: x[1], reverse=True)

    for char, count in sorted_items[:10]:
        if len(char) > 1:  # It's a sequence
            char_display = f"<sequence>{repr(char)}"
        elif char.isspace():
            char_display = f"<space>{ord(char)}"
        elif char == '\n':
            char_display = "<newline>"
        elif char == '\r':
            char_display = "<carriage return>"
        elif char == '\t':
            char_display = "<tab>"
        else:
            char_display = char
        percentage = (count / total_chars) * 100
        click.echo(f"{char_display}: {count:,} ({percentage:.2f}%)")

@cli.command()
@click.argument("patch_file", type=click.Path(exists=True), required=False)
@click.argument("input_dir", type=click.Path(exists=True), required=False)
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory (default: input_dir + '_patched')")
@click.option("--dry-run", is_flag=True, help="Show what would be replaced without making changes")
@click.option("--show-replacements", "-s", is_flag=True, help="Show detailed replacement plan")
@click.option("--create-template", is_flag=True, help="Create a template patch file and exit")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed changes in each file")
def apply_patch(patch_file: str | None, input_dir: str | None, output_dir: str | None,
                dry_run: bool, show_replacements: bool, create_template: bool, verbose: bool) -> None:
    """Apply character replacements from a patch file to VTT files.

    Uses a simple JSON patch file containing character replacements.
    The patch file should contain an array of objects with 'char' and 'replacement' fields.
    Supports both single and multi-character replacements in both directions.

    Example patch file:
    [
        {
            "char": "ú",
            "replacement": "ue",
            "description": "Replace single char with multiple chars"
        },
        {
            "sequence": "...",
            "replacement": "…",
            "description": "Replace multiple chars with single char"
        }
    ]

    Create a template with: asrtk apply-patch --create-template
    """
    if create_template:
        template = [
            {
                "char": "ú",
                "replacement": "ue",
                "description": "Replace acute u with ue"
            },
            {
                "sequence": "</i>",
                "replacement": "",  # Empty string to remove the sequence
                "description": "Remove italic tags"
            },
            {
                "sequence": "<i>",
                "replacement": null,  # null also removes the sequence
                "description": "Remove italic tags"
            },
            {
                "sequence": " ?",
                "replacement": "?",
                "description": "Remove space before question mark"
            }
        ]
        click.echo(json.dumps(template, ensure_ascii=False, indent=2))
        return

    # Validate required arguments when not creating template
    if not patch_file:
        raise click.UsageError("PATCH_FILE is required when not using --create-template")
    if not input_dir:
        raise click.UsageError("INPUT_DIR is required when not using --create-template")

    patch_path = Path(patch_file)
    input_dir = Path(input_dir)

    if not output_dir:
        output_dir = str(input_dir) + '_patched'
    output_dir = Path(output_dir)

    # Load patch file
    with patch_path.open('r', encoding='utf-8') as f:
        patch_data = json.load(f)

    # Create replacements dictionary
    replacements = {}
    for item in patch_data:
        # Handle both char and sequence cases
        key = item.get('char') or item.get('sequence')
        replacement = item.get('replacement')

        if key:
            # Convert "None" string or null to empty string
            if replacement == "None" or replacement is None:
                replacement = ""
            # Skip if replacement is exactly the same as key
            elif replacement == key:
                continue
            replacements[key] = replacement

    if not replacements:
        click.echo("No valid replacements found in patch file")
        return

    click.echo(f"Loaded {len(replacements)} replacements")

    # Show detailed replacement plan if requested
    if show_replacements:
        click.echo("\nPlanned replacements:")
        for old, new in replacements.items():
            desc = next((item.get('description', '') for item in patch_data
                        if item.get('char') == old or item.get('sequence') == old), '')
            click.echo(f"{repr(old)} -> {repr(new)}" + (f" # {desc}" if desc else ""))

        if not click.confirm("\nProceed with these replacements?"):
            click.echo("Operation cancelled")
            return

    # Find all VTT files
    vtt_files = list(input_dir.rglob("*.vtt"))
    if not vtt_files:
        click.echo("No VTT files found in input directory")
        return

    click.echo(f"Found {len(vtt_files)} VTT files")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Process files
    files_changed = 0
    chars_replaced = 0
    console = Console()  # Create console once

    # Don't use progress bar when verbose output is needed
    if verbose or dry_run:
        files_iter = vtt_files
        console.print(f"Processing {len(vtt_files)} files...")
    else:
        files_iter = click.progressbar(vtt_files, label='Processing files')

    try:
        for vtt_file in files_iter:
            try:
                content = vtt_file.read_text(encoding='utf-8')
                new_content = []
                file_chars_replaced = 0
                changes = []

                # Process line by line
                for line in content.split('\n'):
                    new_line = line

                    # Skip timestamp lines
                    if '-->' in line:
                        new_content.append(line)
                        continue

                    # Apply replacements
                    for old_char, new_char in replacements.items():
                        if old_char in new_line:
                            if verbose or dry_run:
                                # Store original before replacement for display
                                if len(changes) < 3:  # Limit to 3 examples
                                    changes.append((new_line, new_line.replace(old_char, new_char),
                                                 old_char, new_char))

                            count = new_line.count(old_char)
                            new_line = new_line.replace(old_char, new_char)
                            file_chars_replaced += count

                    new_content.append(new_line)

                if file_chars_replaced > 0:
                    files_changed += 1
                    chars_replaced += file_chars_replaced

                    # Show changes if requested
                    if dry_run or verbose:
                        console.print(f"\nIn [blue]{vtt_file}[/blue]:")
                        console.print(f"  {file_chars_replaced} replacements")
                        if changes:
                            console.print("  Example changes:")
                            for old, new, old_char, new_char in changes:
                                # Create text objects for better control
                                old_text = Text()
                                new_text = Text()

                                # Split text into parts for highlighting
                                old_parts = old.split(old_char)
                                new_parts = new.split(new_char if new_char else '')

                                # Build highlighted texts
                                for i, part in enumerate(old_parts):
                                    if part:
                                        old_text.append(part)
                                    if i < len(old_parts) - 1:
                                        old_text.append(old_char, style="bold red")

                                for i, part in enumerate(new_parts):
                                    if part:
                                        new_text.append(part)
                                    if i < len(new_parts) - 1 and new_char:
                                        new_text.append(new_char, style="bold green")

                                console.print(f"    Original: {old_text}")
                                console.print(f"    Replace:  {new_text}")
                                console.print(f"    [dim]{repr(old_char)} → {repr(new_char)}[/dim]\n")

                    if not dry_run:
                        # Write the modified content
                        rel_path = vtt_file.relative_to(input_dir)
                        out_file = output_dir / rel_path
                        out_file.parent.mkdir(parents=True, exist_ok=True)
                        out_file.write_text('\n'.join(new_content), encoding='utf-8')

            except Exception as e:
                console.print(f"\n[red]Error processing {vtt_file}: {e}[/red]")
                continue
    finally:
        if not (verbose or dry_run):
            files_iter.finish()  # Clean up progress bar

    if dry_run:
        click.echo(f"\nDry run complete. Would modify {files_changed} files with {chars_replaced} replacements")
    else:
        click.echo(f"\nProcessed {len(vtt_files)} files")
        click.echo(f"Modified {files_changed} files")
        click.echo(f"Made {chars_replaced} character replacements")
        click.echo(f"Results saved to {output_dir}")

@cli.command()
@click.argument("work_dir", type=click.Path(exists=True))
@click.argument("words", nargs=-1, required=True)
@click.option("--output", "-o", type=str, default="word_matches.txt", help="Output text file name")
@click.option("--context", "-c", type=int, default=50, help="Number of characters for context (default: 50)")
@click.option("--ignore-case", "-i", is_flag=True, help="Case insensitive search")
@click.option("--whole-word", "-w", is_flag=True, help="Match whole words only")
@click.option("--regex", "-r", is_flag=True, help="Treat search terms as regular expressions")
def find_words(work_dir: str, words: tuple[str, ...], output: str, context: int,
               ignore_case: bool, whole_word: bool, regex: bool) -> None:
    """Search for specific words or phrases in VTT files.

    Searches recursively through all VTT files in the directory for the given words
    and outputs matches with context to a file.

    Examples:
        asrtk find-words ./subtitles "hello" "world"
        asrtk find-words ./subtitles "error" -i -w
        asrtk find-words ./subtitles "h[ae]llo" -r
    """
    work_dir = Path(work_dir)

    # Find all VTT files recursively
    vtt_files = list(work_dir.rglob("*.vtt"))

    if not vtt_files:
        click.echo("No VTT files found in the work directory")
        return

    click.echo(f"Found {len(vtt_files)} VTT files")

    # Prepare search patterns
    import re
    patterns = []
    for word in words:
        if regex:
            pattern = word
        else:
            pattern = re.escape(word)
            if whole_word:
                pattern = fr'\b{pattern}\b'

        try:
            patterns.append(re.compile(pattern, re.IGNORECASE if ignore_case else 0))
        except re.error as e:
            click.echo(f"Invalid pattern '{word}': {e}", err=True)
            return

    # Process files
    matches = []
    total_matches = 0

    with click.progressbar(vtt_files, label='Searching files') as files:
        for vtt_file in files:
            try:
                content = vtt_file.read_text(encoding='utf-8')
                file_matches = []

                # Skip VTT header and timestamp lines
                for line in content.split('\n'):
                    if line.strip() and not line.startswith('WEBVTT') and '-->' not in line:
                        for pattern in patterns:
                            for match in pattern.finditer(line):
                                total_matches += 1
                                # Get context around match
                                start = max(0, match.start() - context)
                                end = min(len(line), match.end() + context)

                                match_info = {
                                    'file': str(vtt_file),
                                    'pattern': pattern.pattern,
                                    'matched_text': match.group(),
                                    'context': line[start:end],
                                    'start_pos': start,
                                    'match_pos': match.start() - start,
                                    'match_len': match.end() - match.start()
                                }
                                file_matches.append(match_info)

                if file_matches:
                    matches.extend(file_matches)

            except Exception as e:
                click.echo(f"\nError processing {vtt_file}: {e}", err=True)
                continue

    if not matches:
        click.echo("No matches found")
        return

    # Save results
    output_file = Path(output)
    with output_file.open('w', encoding='utf-8') as f:
        f.write(f"Found {total_matches} matches in {len(vtt_files)} files\n")
        f.write("Search parameters:\n")
        f.write(f"  Words: {', '.join(words)}\n")
        f.write(f"  Ignore case: {ignore_case}\n")
        f.write(f"  Whole word: {whole_word}\n")
        f.write(f"  Regex: {regex}\n\n")

        current_file = None
        for match in matches:
            # Print filename when it changes
            if current_file != match['file']:
                current_file = match['file']
                f.write(f"\nFile: {current_file}\n")
                f.write("-" * 80 + "\n")

            # Write match with context
            context = match['context']
            pos = match['match_pos']
            length = match['match_len']

            # Highlight the match using brackets
            highlighted = (
                context[:pos] +
                '[' + context[pos:pos+length] + ']' +
                context[pos+length:]
            )

            f.write(f"Match: {match['pattern']}\n")
            f.write(f"Context: {highlighted}\n\n")

    click.echo(f"\nFound {total_matches} matches")
    click.echo(f"Results saved to: {output_file}")

@cli.command()
@click.argument("work_dir", type=click.Path(exists=True))
@click.option("--output", "-o", type=str, default="arabic_subs.txt", help="Output text file name")
@click.option("--batch", "-b", is_flag=True, help="Create batch file for deletion")
def find_arabic(work_dir: str, output: str, batch: bool) -> None:
    """Find VTT files containing Arabic characters.

    Creates a text file listing all VTT files that contain Arabic text.
    With --batch flag, creates a batch file with del commands.
    """
    work_dir = Path(work_dir)

    # Find all VTT files recursively
    vtt_files = list(work_dir.rglob("*.vtt"))

    if not vtt_files:
        click.echo("No VTT files found in the work directory")
        return

    click.echo(f"Found {len(vtt_files)} VTT files")

    # Arabic Unicode ranges
    def has_arabic(text: str) -> bool:
        return any(ord(char) in range(0x0600, 0x06FF) or  # Arabic
                  ord(char) in range(0xFE70, 0xFEFF) or   # Arabic Presentation Forms-B
                  ord(char) in range(0xFB50, 0xFDFF)      # Arabic Presentation Forms-A
                  for char in text)

    # Process files
    arabic_files = []

    with click.progressbar(vtt_files, label='Scanning files') as files:
        for vtt_file in files:
            try:
                content = vtt_file.read_text(encoding='utf-8')
                if has_arabic(content):
                    arabic_files.append(vtt_file)
            except Exception as e:
                click.echo(f"\nError processing {vtt_file}: {e}", err=True)
                continue

    if not arabic_files:
        click.echo("No VTT files with Arabic characters found")
        return

    # Save results
    output_file = Path(output)
    batch_file = output_file.with_suffix('.bat') if batch else None

    with output_file.open('w', encoding='utf-8') as f:
        for file in arabic_files:
            f.write(f"{file}\n")

    if batch:
        with batch_file.open('w', encoding='utf-8') as f:
            f.write("@echo off\n")
            f.write("echo Deleting VTT files containing Arabic characters...\n")
            for file in arabic_files:
                # Use quotes to handle paths with spaces
                f.write(f'del "{file}"\n')
            f.write("echo Done.\n")
            f.write("pause\n")

    click.echo(f"\nFound {len(arabic_files)} files containing Arabic characters")
    click.echo(f"File list saved to: {output_file}")
    if batch:
        click.echo(f"Batch file saved to: {batch_file}")

def find_sample_sentences(text: str, char: str, max_samples: int = 3, max_length: int = 100) -> list[str]:
    """Find sample sentences containing the given character.

    Args:
        text: The text to search in
        char: The character to find
        max_samples: Maximum number of samples to return
        max_length: Maximum length of each sample

    Returns:
        List of sample sentences/contexts
    """
    samples = []
    lines = text.split('\n')

    for line in lines:
        if char in line and not line.startswith('WEBVTT') and '-->' not in line:
            # Trim long lines
            if len(line) > max_length:
                # Find the char position
                pos = line.find(char)
                # Take some context before and after
                start = max(0, pos - max_length//2)
                end = min(len(line), pos + max_length//2)
                sample = ('...' if start > 0 else '') + \
                        line[start:end] + \
                        ('...' if end < len(line) else '')
            else:
                sample = line

            if sample not in samples:  # Avoid duplicates
                samples.append(sample)
                if len(samples) >= max_samples:
                    break

    return samples

def clean_caption_text(text: str) -> str:
    """Clean caption text by:
    1. Strip whitespace
    2. Remove trailing hyphens
    3. Strip again
    4. Normalize spaces
    5. Apply sanitize function

    Args:
        text: Raw caption text
    Returns:
        Cleaned text
    """
    text = text.strip()
    while text.endswith('-'):
        text = text[:-1]
    text = text.strip()
    # Replace multiple spaces with single space
    while '  ' in text:
        text = text.replace('  ', ' ')
    return text

@cli.command()
@click.argument("work_dir", type=click.Path(exists=True))
@click.option("--output", "-o", type=str, default="wordset.json", help="Output JSON file name")
@click.option("--min-frequency", "-f", type=int, default=1, help="Minimum word frequency (default: 1)")
@click.option("--ignore-case", "-i", is_flag=True, help="Case insensitive word counting")
@click.option("--turkish", "-t", is_flag=True, help="Use Turkish-specific word processing")
def create_wordset(work_dir: str, output: str, min_frequency: int, ignore_case: bool, turkish: bool) -> None:
    """Create word frequency statistics from VTT files."""
    from .utils import (
        get_unique_words_with_frequencies,
        turkish_lower,
        read_vtt_as_text,
        sanitize
    )
    import webvtt  # Use webvtt directly for better control

    work_dir = Path(work_dir)
    vtt_files = list(work_dir.rglob("*.vtt"))

    if not vtt_files:
        click.echo("No VTT files found in the work directory")
        return

    click.echo(f"Found {len(vtt_files)} VTT files")

    # Read and clean all files
    click.echo("Reading files...")
    all_text = []
    with click.progressbar(vtt_files, label='Reading files') as files:
        for vtt_file in files:
            try:
                # Read VTT file properly
                captions = webvtt.read(str(vtt_file))
                # Get only the text content and clean each caption
                for caption in captions:
                    # Clean text in stages
                    text = clean_caption_text(caption.text)
                    text = sanitize(text)  # Apply utils.sanitize after basic cleaning
                    if ignore_case:
                        text = turkish_lower(text) if turkish else text.lower()
                    if text.strip():  # Only add non-empty lines
                        all_text.append(text)
            except Exception as e:
                click.echo(f"\nError reading {vtt_file}: {e}", err=True)
                continue

    # Process all text at once
    click.echo("Processing text...")
    corpus = " ".join(all_text)
    words, frequencies = get_unique_words_with_frequencies(corpus)
    total_words = sum(frequencies.values())

    # Prepare JSON structure
    wordset_data = {
        "stats": {
            "total_words": total_words,
            "unique_words": len(frequencies),
            "min_frequency": min_frequency,
            "ignore_case": ignore_case,
            "turkish_mode": turkish
        },
        "words": []
    }

    # Add words
    for word, count in sorted(frequencies.items(), key=lambda x: x[1], reverse=True):
        if count >= min_frequency:
            word_info = {
                "word": word,
                "frequency": count,
                "percentage": (count / total_words) * 100
            }
            wordset_data["words"].append(word_info)

    # Save to JSON file
    output_file = Path(output)
    with output_file.open('w', encoding='utf-8') as f:
        json.dump(wordset_data, f, ensure_ascii=False, indent=2)

    click.echo(f"\nProcessed {total_words:,} total words")
    click.echo(f"Found {len(frequencies):,} unique words")
    click.echo(f"Results saved to {output_file}")

    # Display top 10 most frequent words
    click.echo("\nTop 10 most frequent words:")
    for word, count in sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:10]:
        percentage = (count / total_words) * 100
        click.echo(f"{word}: {count:,} ({percentage:.2f}%)")

@cli.command()
@click.argument("work_dir", type=click.Path(exists=True))
@click.option("--output", "-o", type=str, default="pattern_matches.txt", help="Output text file name")
@click.option("--context", "-c", type=int, default=50, help="Number of characters for context (default: 50)")
@click.option("--pattern", "-p", multiple=True, help="Custom regex patterns to search for")
@click.option("--numbers", "-n", is_flag=True, help="Search for potentially incorrect number formats")
def find_patterns(work_dir: str, output: str, context: int, pattern: tuple[str, ...], numbers: bool) -> None:
    """Search for specific patterns in VTT files.

    Specialized pattern matching for finding formatting issues like incorrect number formats.
    Particularly useful for finding English-style numbers in Turkish text (comma as thousands separator).

    Built-in patterns when using --numbers:
    - Numbers with comma followed by 3+ digits (likely English format)
    - Numbers with multiple commas (e.g., 1,234,567)

    Examples:
        # Find potentially incorrect number formats
        asrtk find-patterns ./subtitles --numbers

        # Search with custom pattern
        asrtk find-patterns ./subtitles -p "\d+,\d{3,}"

        # Multiple patterns
        asrtk find-patterns ./subtitles -p "\d+,\d{3,}" -p "\d+\.\d{3,}"
    """
    work_dir = Path(work_dir)

    # Predefined patterns for number formats
    number_patterns = [
        (r"\d+,\d{3,}", "Number with comma followed by 3+ digits (possible English format)"),
        (r"\d+(?:,\d+){2,}", "Number with multiple commas (e.g., 1,234,567)"),  # Updated pattern
        (r"(?<!\d)[.,]\d+", "Number starting with decimal point/comma"),
        (r"\d{1,3}(?:,\d{3})+(?!\d)", "Standard English number format (e.g., 1,234 or 1,234,567)"),
        (r"\d+,\d+,", "Number with consecutive commas"),  # Catch malformed numbers
    ]

    # Compile all patterns
    compiled_patterns = []

    # Add custom patterns
    for p in pattern:
        try:
            compiled_patterns.append((re.compile(p), f"Custom pattern: {p}"))
        except re.error as e:
            click.echo(f"Invalid pattern '{p}': {e}", err=True)
            return

    # Add number patterns if requested
    if numbers:
        for p, desc in number_patterns:
            compiled_patterns.append((re.compile(p), desc))

    if not compiled_patterns:
        click.echo("No patterns specified. Use --numbers or provide patterns with -p")
        return

    # Find all VTT files
    vtt_files = list(work_dir.rglob("*.vtt"))
    if not vtt_files:
        click.echo("No VTT files found in the work directory")
        return

    click.echo(f"Found {len(vtt_files)} VTT files")

    # Process files
    matches = []
    total_matches = 0
    console = Console()

    with click.progressbar(vtt_files, label='Searching files') as files:
        for vtt_file in files:
            try:
                content = vtt_file.read_text(encoding='utf-8')
                file_matches = []

                # Skip VTT header and timestamp lines
                for line in content.split('\n'):
                    if line.strip() and not line.startswith('WEBVTT') and '-->' not in line:
                        for pattern, desc in compiled_patterns:
                            for match in pattern.finditer(line):
                                total_matches += 1
                                # Get context around match
                                start = max(0, match.start() - context)
                                end = min(len(line), match.end() + context)

                                match_info = {
                                    'file': str(vtt_file),
                                    'pattern': pattern.pattern,
                                    'description': desc,
                                    'matched_text': match.group(),
                                    'context': line[start:end],
                                    'start_pos': start,
                                    'match_pos': match.start() - start,
                                    'match_len': match.end() - match.start()
                                }
                                file_matches.append(match_info)

                if file_matches:
                    matches.extend(file_matches)

            except Exception as e:
                click.echo(f"\nError processing {vtt_file}: {e}", err=True)
                continue

    if not matches:
        click.echo("No matches found")
        return

    # Save results
    output_file = Path(output)
    with output_file.open('w', encoding='utf-8') as f:
        f.write(f"Found {total_matches} matches in {len(vtt_files)} files\n")
        f.write("Search patterns:\n")
        for _, desc in compiled_patterns:
            f.write(f"  - {desc}\n")
        f.write("\n")

        current_file = None
        for match in matches:
            # Print filename when it changes
            if current_file != match['file']:
                current_file = match['file']
                f.write(f"\nFile: {current_file}\n")
                f.write("-" * 80 + "\n")

            # Write match with context
            context = match['context']
            pos = match['match_pos']
            length = match['match_len']

            # Highlight the match using brackets
            highlighted = (
                context[:pos] +
                '[' + context[pos:pos+length] + ']' +
                context[pos+length:]
            )

            f.write(f"Pattern: {match['pattern']} ({match['description']})\n")
            f.write(f"Found: {match['matched_text']}\n")
            f.write(f"Context: {highlighted}\n\n")

    # Display summary with rich formatting
    console.print(f"\nFound [bold]{total_matches}[/bold] matches")
    console.print(f"Results saved to: [blue]{output_file}[/blue]")

    # Show sample of matches
    if matches:
        console.print("\nSample matches:")
        for match in matches[:5]:
            text = Text()
            context = match['context']
            pos = match['match_pos']
            length = match['match_len']

            text.append(context[:pos])
            text.append(context[pos:pos+length], style="bold red")
            text.append(context[pos+length:])

            console.print(f"  [cyan]{match['matched_text']}[/cyan] in context:")
            console.print(f"  {text}")

@cli.command()
@click.argument("work_dir", type=click.Path(exists=True))
@click.option("--output", "-o", type=str, default="brackets.json", help="Output JSON file name")
@click.option("--min-frequency", "-f", type=int, default=1, help="Minimum frequency to include (default: 1)")
@click.option("--ignore-case", "-i", is_flag=True, help="Case insensitive counting")
@click.option("--show-context", "-c", is_flag=True, help="Include sample contexts in output")
def find_brackets(work_dir: str, output: str, min_frequency: int, ignore_case: bool, show_context: bool) -> None:
    """Find and analyze text within brackets and parentheses in VTT files.

    Creates a frequency list of text found within () and [] with their occurrences.
    Useful for finding annotations, sound effects, or other bracketed content.

    Examples:
        # Basic usage
        asrtk find-brackets ./subtitles

        # Case insensitive with minimum frequency
        asrtk find-brackets ./subtitles -i -f 2

        # Include context samples
        asrtk find-brackets ./subtitles -c
    """
    from collections import defaultdict
    import re

    work_dir = Path(work_dir)

    # Compile regex patterns
    patterns = {
        'parentheses': (r'\((.*?)\)', 'Round brackets ()'),
        'square': (r'\[(.*?)\]', 'Square brackets []')
    }
    compiled_patterns = {
        name: (re.compile(pattern), desc)
        for name, (pattern, desc) in patterns.items()
    }

    # Initialize counters and context storage
    bracket_counts = defaultdict(lambda: defaultdict(int))
    bracket_contexts = defaultdict(lambda: defaultdict(list))
    total_matches = 0

    # Find all VTT files
    vtt_files = list(work_dir.rglob("*.vtt"))
    if not vtt_files:
        click.echo("No VTT files found in the work directory")
        return

    click.echo(f"Found {len(vtt_files)} VTT files")

    # Process files
    with click.progressbar(vtt_files, label='Processing files') as files:
        for vtt_file in files:
            try:
                content = vtt_file.read_text(encoding='utf-8')

                # Process each line
                for line in content.split('\n'):
                    if line.strip() and not line.startswith('WEBVTT') and '-->' not in line:
                        # Check each pattern
                        for name, (pattern, _) in compiled_patterns.items():
                            matches = pattern.finditer(line)
                            for match in matches:
                                text = match.group(1).strip()  # Get text inside brackets
                                if not text:
                                    continue

                                if ignore_case:
                                    text = text.lower()

                                bracket_counts[name][text] += 1
                                total_matches += 1

                                # Store context if needed (limit to 3 examples per text)
                                if show_context and len(bracket_contexts[name][text]) < 3:
                                    # Get some context around the match
                                    start = max(0, match.start() - 30)
                                    end = min(len(line), match.end() + 30)
                                    context = line[start:end].strip()
                                    if context not in bracket_contexts[name][text]:
                                        bracket_contexts[name][text].append(context)

            except Exception as e:
                click.echo(f"\nError processing {vtt_file}: {e}", err=True)
                continue

    # Prepare output data
    output_data = {
        "stats": {
            "total_matches": total_matches,
            "min_frequency": min_frequency,
            "ignore_case": ignore_case,
            "files_processed": len(vtt_files)
        },
        "brackets": {}
    }

    # Process each bracket type
    for bracket_type, counts in bracket_counts.items():
        entries = []
        for text, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            if count >= min_frequency:
                entry = {
                    "text": text,
                    "frequency": count,
                    "percentage": (count / total_matches) * 100
                }
                if show_context and text in bracket_contexts[bracket_type]:
                    entry["samples"] = bracket_contexts[bracket_type][text]
                entries.append(entry)

        output_data["brackets"][bracket_type] = {
            "description": compiled_patterns[bracket_type][1],
            "total": sum(counts.values()),
            "unique": len(entries),
            "entries": entries
        }

    # Save to JSON file
    output_file = Path(output)
    with output_file.open('w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # Display summary
    console = Console()
    console.print(f"\nFound [bold]{total_matches}[/bold] total bracketed texts")

    for bracket_type, data in output_data["brackets"].items():
        console.print(f"\n[blue]{data['description']}[/blue]:")
        console.print(f"  Total occurrences: {data['total']}")
        console.print(f"  Unique texts: {data['unique']}")

        # Show top 5 most frequent
        console.print("\n  Most frequent:")
        for entry in data['entries'][:5]:
            console.print(f"    [cyan]{entry['text']}[/cyan]: {entry['frequency']} times")
            if show_context and "samples" in entry:
                for sample in entry['samples']:
                    text = Text(sample)
                    text.highlight_words([entry['text']], style="bold red")
                    console.print(f"      {text}")

    console.print(f"\nFull results saved to: [blue]{output_file}[/blue]")
