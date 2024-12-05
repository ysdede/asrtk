"""Command for creating word frequency statistics from VTT files."""
from pathlib import Path
import rich_click as click
import webvtt
from typing import List
import json

from ...core.text import (
    get_unique_words_with_frequencies,
    turkish_lower,
    sanitize
)
from ...core.vtt import clean_caption_text

@click.command()
@click.argument("work_dir", type=click.Path(exists=True))
@click.option("--output", "-o", type=str, default="wordset.json", help="Output JSON file name")
@click.option("--min-frequency", "-f", type=int, default=1, help="Minimum word frequency (default: 1)")
@click.option("--ignore-case", "-i", is_flag=True, help="Case insensitive word counting")
@click.option("--turkish", "-t", is_flag=True, help="Use Turkish-specific word processing")
def create_wordset(work_dir: str, output: str, min_frequency: int, ignore_case: bool, turkish: bool) -> None:
    """Create word frequency statistics from VTT files."""
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
