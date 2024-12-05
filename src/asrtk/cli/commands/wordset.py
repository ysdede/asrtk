"""Command for creating word frequency statistics from VTT files."""
from pathlib import Path
import rich_click as click
from typing import List
import json
from collections import Counter
from rich.console import Console
from rich.table import Table

from ...core.text import (
    get_unique_words_with_frequencies,
    turkish_lower,
    sanitize
)
from ...core.vtt import read_vtt_file

console = Console()

@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--output', '-o', type=str, default="wordset.txt", help="Output text file name")
@click.option('--min-frequency', '-f', type=int, default=1, help="Minimum word frequency to include")
@click.option('--ignore-case', '-i', is_flag=True, help="Case insensitive counting")
@click.option('--turkish', '-t', is_flag=True, help="Use Turkish-specific word processing")
def create_wordset(input_dir: str, output: str, min_frequency: int, ignore_case: bool, turkish: bool) -> None:
    """Create word frequency statistics from VTT files.

    Processes all VTT files in the directory recursively and creates a frequency-sorted
    word list. Useful for creating pronunciation dictionaries or analyzing vocabulary.

    Examples:
        # Basic usage
        asrtk create-wordset ./subtitles

        # Case insensitive with minimum frequency
        asrtk create-wordset ./subtitles -i -f 2

        # Use Turkish-specific processing
        asrtk create-wordset ./subtitles -t
    """
    input_path = Path(input_dir)

    # Find all VTT files recursively
    vtt_files = list(input_path.rglob("*.vtt"))
    if not vtt_files:
        console.print("No VTT files found in the input directory")
        return

    console.print(f"Found {len(vtt_files)} VTT files")

    # Process files and collect words
    word_counts = Counter()
    total_words = 0
    files_processed = 0

    with console.status("[bold green]Processing files...") as status:
        for vtt_file in vtt_files:
            try:
                # Get text content from VTT file using common utility
                text_lines = read_vtt_file(vtt_file)

                # Process each line
                for line in text_lines:
                    # Clean and normalize text
                    clean_text = sanitize(line)
                    if ignore_case:
                        clean_text = turkish_lower(clean_text) if turkish else clean_text.lower()

                    # Use proper word extraction
                    words, _ = get_unique_words_with_frequencies(clean_text)
                    if words:  # Only update if we found valid words
                        word_counts.update(words)
                        total_words += len(words)

                files_processed += 1
                status.update(f"[bold green]Processing files... {files_processed}/{len(vtt_files)}")

            except Exception as e:
                console.print(f"[red]Error processing {vtt_file}: {e}[/red]")
                continue

    # Filter by minimum frequency and sort by frequency (descending)
    sorted_words = [(word, count) for word, count in word_counts.items()
                   if count >= min_frequency]
    sorted_words.sort(key=lambda x: (-x[1], x[0]))  # Sort by frequency desc, then word asc

    if not sorted_words:
        console.print("No words found meeting the minimum frequency requirement")
        return

    # Save results with frequencies
    output_file = Path(output)
    with output_file.open('w', encoding='utf-8') as f:
        f.write(f"# Word frequency list from {len(vtt_files)} VTT files\n")
        f.write(f"# Files processed: {files_processed}\n")
        f.write(f"# Total words processed: {total_words:,}\n")
        f.write(f"# Unique words (freq >= {min_frequency}): {len(sorted_words):,}\n")
        f.write(f"# Case sensitive: {not ignore_case}\n")
        f.write(f"# Turkish mode: {turkish}\n\n")
        f.write("# Format: word<tab>frequency<tab>percentage\n\n")

        # Always write frequencies and percentages
        for word, count in sorted_words:
            percentage = (count / total_words) * 100
            f.write(f"{word}\t{count}\t{percentage:.4f}%\n")

    # Display summary
    console.print(f"\n[green]Processed {files_processed:,} files")
    console.print(f"Total words processed: {total_words:,}")
    console.print(f"Found {len(sorted_words):,} unique words with frequency >= {min_frequency}")
    console.print(f"Results saved to: [blue]{output_file}[/blue]")

    # Show most frequent words in a table
    table = Table(title="Most Frequent Words")
    table.add_column("Word", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("Percentage", justify="right", style="yellow")

    for word, count in sorted_words[:20]:  # Show top 20 words
        percentage = (count / total_words) * 100
        table.add_row(word, f"{count:,}", f"{percentage:.2f}%")

    console.print("\n", table)
