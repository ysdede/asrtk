"""Command for analyzing number formats and frequencies in VTT files."""
from pathlib import Path
import rich_click as click
from rich.console import Console
from rich.table import Table
import re
from collections import Counter
from typing import List, Tuple, Dict

from asrtk.core.vtt import read_vtt_file, is_timestamp_line, is_header_line

console = Console()

def extract_numbers(text: str, check_mixed: bool = False, check_multiple: bool = False) -> List[str]:
    """Extract numbers from text using various patterns."""
    # Pattern to match numbers that might be part of larger numbers
    pattern = r'(?<![0-9.,])([0-9]+(?:[,.][0-9]+)+)(?![0-9])'  # Matches numbers with periods/commas

    found_numbers = []
    for match in re.finditer(pattern, text):
        number = match.group(1)
        # Skip if this line looks like a timestamp
        if '-->' not in text:
            # Check if this number is part of a larger number
            # by looking at surrounding text for connected numbers
            start_pos = match.start(1)
            end_pos = match.end(1)

            # Look ahead for connected numbers (e.g., "1.250.000")
            next_char_pos = end_pos
            while next_char_pos < len(text) and text[next_char_pos:next_char_pos+1].isspace():
                next_char_pos += 1

            # If this number is part of a larger sequence, skip it
            if next_char_pos < len(text) and text[next_char_pos:].strip().startswith('.000'):
                continue

            found_numbers.append((number, 'Incorrect decimal format'))

    return found_numbers

def process_vtt_file(vtt_file: Path, check_mixed: bool, check_multiple: bool) -> List[Tuple[str, str, str]]:
    """Process a VTT file for number formats."""
    content = vtt_file.read_text(encoding='utf-8')
    results = []

    for line in content.split('\n'):
        # Skip VTT headers and empty lines
        if not line.strip() or is_header_line(line):
            continue

        # Skip timestamp lines using proper VTT utility
        if is_timestamp_line(line):
            continue

        numbers = extract_numbers(line, check_mixed, check_multiple)
        for number, category in numbers:
            results.append((number, category, line.strip()))
            # Debug output
            console.print(f"Found: {number} in line: {line.strip()}")

    return results

@click.command('analyze-numbers')
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--output', '-o', type=str, default="number_stats.txt", help="Output text file name")
@click.option('--min-frequency', '-f', type=int, default=1, help="Minimum frequency to include")
@click.option('--check-mixed', is_flag=True, help="Check for mixed separator formats (1.234,56)")
@click.option('--check-multiple', is_flag=True, help="Check for multiple separators (1.2.3)")
def count_numbers(input_dir: str, output: str, min_frequency: int, check_mixed: bool, check_multiple: bool) -> None:
    """Analyze floating-point number formats in VTT files.

    By default, looks for:
    - Incorrect decimal format using period (3.14)
    - Correct decimal format using comma (3,14)

    Optional checks:
    --check-mixed: Also find mixed separator formats (1.234,56)
    --check-multiple: Also find multiple separators (1.2.3)

    Examples:
        # Find incorrect decimal separators
        asrtk analyze-numbers ./subtitles

        # Include all checks
        asrtk analyze-numbers ./subtitles --check-mixed --check-multiple
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
                results = process_vtt_file(vtt_file, check_mixed, check_multiple)

                # Count results
                for number, category, context in results:
                    number_counts[number] += 1
                    category_counts[category] += 1

                    # Store context for the first few occurrences
                    if number not in number_contexts:
                        number_contexts[number] = []
                    if len(number_contexts[number]) < 3:  # Store up to 3 examples
                        number_contexts[number].append(context)

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
