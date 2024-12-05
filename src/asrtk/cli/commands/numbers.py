"""Command for analyzing number formats and frequencies in VTT files."""
from pathlib import Path
import rich_click as click
from rich.console import Console
from rich.table import Table
import re
from collections import Counter
from typing import List, Tuple, Dict

from ...core.vtt import read_vtt_file

console = Console()

def extract_numbers(text: str) -> List[str]:
    """Extract numbers from text using various patterns.

    Looks for:
    - Numbers with period as thousands separator (incorrect in Turkish)
    - Numbers with comma as decimal separator (correct in Turkish)
    - Numbers with multiple separators
    - Plain numbers
    """
    patterns = [
        # Incorrect format for Turkish (period as thousands separator)
        (r'\b\d{1,3}(?:\.\d{3})+(?:,\d+)?\b', 'Period as thousands separator'),

        # Correct Turkish format (comma as decimal)
        (r'\b\d+,\d+\b', 'Comma as decimal separator'),

        # Multiple separators (potentially problematic)
        (r'\b\d+[.,]\d+[.,]\d+\b', 'Multiple separators'),

        # Plain numbers (for reference)
        (r'\b\d+\b', 'Plain number')
    ]

    found_numbers = []
    for pattern, category in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            found_numbers.append((match.group(), category))

    return found_numbers

@click.command('count-numbers')
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--output', '-o', type=str, default="number_stats.txt", help="Output text file name")
@click.option('--min-frequency', '-f', type=int, default=1, help="Minimum frequency to include")
def count_numbers(input_dir: str, output: str, min_frequency: int) -> None:
    """Analyze number formats and their frequencies in VTT files.

    This command processes VTT files and analyzes number formats, focusing on:
    - Numbers with period as thousands separator (incorrect in Turkish)
    - Numbers with comma as decimal separator (correct in Turkish)
    - Numbers with multiple separators (potentially problematic)
    - Plain numbers

    Examples:
        # Basic usage
        asrtk count-numbers ./subtitles

        # With minimum frequency
        asrtk count-numbers ./subtitles -f 2
    """
    input_path = Path(input_dir)

    # Find all VTT files
    vtt_files = list(input_path.rglob("*.vtt"))
    if not vtt_files:
        console.print("No VTT files found in input directory")
        return

    console.print(f"Found {len(vtt_files)} VTT files")

    # Process files and collect numbers
    number_counts = Counter()
    category_counts = Counter()
    number_contexts: Dict[str, List[str]] = {}
    files_processed = 0

    with console.status("[bold green]Processing files...") as status:
        for vtt_file in vtt_files:
            try:
                text_lines = read_vtt_file(vtt_file)

                for line in text_lines:
                    numbers = extract_numbers(line)
                    for number, category in numbers:
                        number_counts[number] += 1
                        category_counts[category] += 1

                        # Store context for the first few occurrences
                        if number not in number_contexts:
                            number_contexts[number] = []
                        if len(number_contexts[number]) < 3:  # Store up to 3 examples
                            number_contexts[number].append(line.strip())

                files_processed += 1
                status.update(f"[bold green]Processing files... {files_processed}/{len(vtt_files)}")

            except Exception as e:
                console.print(f"[red]Error processing {vtt_file}: {e}[/red]")
                continue

    # Filter and sort numbers by frequency
    sorted_numbers = [(num, count) for num, count in number_counts.items()
                     if count >= min_frequency]
    sorted_numbers.sort(key=lambda x: (-x[1], x[0]))

    # Save results
    output_file = Path(output)
    with output_file.open('w', encoding='utf-8') as f:
        f.write(f"# Number format analysis from {len(vtt_files)} VTT files\n")
        f.write(f"# Files processed: {files_processed}\n")
        f.write(f"# Total unique numbers: {len(sorted_numbers)}\n\n")

        # Write category statistics
        f.write("# Number format categories:\n")
        for category, count in category_counts.most_common():
            f.write(f"# {category}: {count:,}\n")
        f.write("\n")

        f.write("# Format: number<tab>frequency<tab>example_context\n\n")

        for number, count in sorted_numbers:
            contexts = number_contexts.get(number, [])
            context_str = " | ".join(contexts[:1])  # Show first context only in file
            f.write(f"{number}\t{count}\t{context_str}\n")

    # Display summary
    console.print(f"\n[green]Processed {files_processed:,} files")
    console.print(f"Found {len(sorted_numbers):,} unique numbers")
    console.print(f"Results saved to: [blue]{output_file}[/blue]")

    # Show category statistics in a table
    cat_table = Table(title="Number Format Categories")
    cat_table.add_column("Category", style="cyan")
    cat_table.add_column("Count", justify="right", style="green")

    for category, count in category_counts.most_common():
        cat_table.add_row(category, f"{count:,}")

    console.print("\n", cat_table)

    # Show most frequent numbers in a table
    num_table = Table(title="Most Frequent Numbers")
    num_table.add_column("Number", style="cyan")
    num_table.add_column("Count", justify="right", style="green")
    num_table.add_column("Example Context", style="yellow")

    for number, count in sorted_numbers[:20]:  # Show top 20 numbers
        context = number_contexts.get(number, [""])[0]
        num_table.add_row(number, f"{count:,}", context)

    console.print("\n", num_table)
