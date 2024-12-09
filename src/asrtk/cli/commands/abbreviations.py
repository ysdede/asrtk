"""Command for finding abbreviation patterns in VTT files."""
from pathlib import Path
import rich_click as click
from rich.console import Console
from rich.table import Table
import re
from collections import Counter
from typing import List, Dict

from ...core.vtt import read_vtt_file

console = Console()

def extract_abbreviations(text: str) -> List[tuple[str, str]]:
    """Extract abbreviation patterns from text.

    Looks for patterns like:
    - S.A.
    - S.A:
    - A.Ş.
    etc.

    Args:
        text: Text to analyze

    Returns:
        List of tuples (abbreviation, category)
    """
    patterns = [
        # Pattern with periods between letters and colon at end (S.A:)
        (r'\b[A-Z]\.[A-Z]:', 'Letter.Letter:'),

        # Pattern with periods between letters and period at end (S.A.)
        (r'\b[A-Z]\.[A-Z]\.', 'Letter.Letter.'),

        # Pattern with periods between letters (S.A)
        (r'\b[A-Z]\.[A-Z]\b', 'Letter.Letter'),

        # Pattern with three letters (A.Ş.A.)
        (r'\b[A-Z]\.[A-ZŞĞÜÇİÖ]\.[A-Z]\.', 'Letter.Letter.Letter.'),

        # Pattern with Turkish characters (A.Ş.)
        (r'\b[A-Z]\.[ŞĞÜÇİÖ]\.', 'Letter.TurkishLetter.'),
    ]

    found_patterns = []
    for pattern, category in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            found_patterns.append((match.group(), category))

    return found_patterns

@click.command('find-abbreviations')
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--output', '-o', type=str, default="abbreviations.txt", help="Output text file name")
@click.option('--min-frequency', '-f', type=int, default=1, help="Minimum frequency to include")
def find_abbreviations(input_dir: str, output: str, min_frequency: int) -> None:
    """Find and analyze abbreviation patterns in VTT files.

    This command looks for patterns like:
    - S.A.  (Letter.Letter.)
    - S.A:  (Letter.Letter:)
    - A.Ş.  (Letter.TurkishLetter.)

    Examples:
        # Basic usage
        asrtk find-abbreviations ./subtitles

        # With minimum frequency
        asrtk find-abbreviations ./subtitles -f 2
    """
    input_path = Path(input_dir)

    # Find all VTT files
    vtt_files = list(input_path.rglob("*.vtt"))
    if not vtt_files:
        console.print("No VTT files found in input directory")
        return

    console.print(f"Found {len(vtt_files)} VTT files")

    # Process files
    pattern_counts = Counter()
    category_counts = Counter()
    pattern_contexts: Dict[str, List[str]] = {}
    files_processed = 0

    with console.status("[bold green]Processing files...") as status:
        for vtt_file in vtt_files:
            try:
                text_lines = read_vtt_file(vtt_file)

                for line in text_lines:
                    patterns = extract_abbreviations(line)
                    for pattern, category in patterns:
                        pattern_counts[pattern] += 1
                        category_counts[category] += 1

                        # Store context for the first few occurrences
                        if pattern not in pattern_contexts:
                            pattern_contexts[pattern] = []
                        if len(pattern_contexts[pattern]) < 3:  # Store up to 3 examples
                            pattern_contexts[pattern].append(line.strip())

                files_processed += 1
                status.update(f"[bold green]Processing files... {files_processed}/{len(vtt_files)}")

            except Exception as e:
                console.print(f"[red]Error processing {vtt_file}: {e}[/red]")
                continue

    # Filter and sort patterns by frequency
    sorted_patterns = [(pat, count) for pat, count in pattern_counts.items()
                      if count >= min_frequency]
    sorted_patterns.sort(key=lambda x: (-x[1], x[0]))

    # Save results
    output_file = Path(output)
    with output_file.open('w', encoding='utf-8') as f:
        f.write(f"# Abbreviation pattern analysis from {len(vtt_files)} VTT files\n")
        f.write(f"# Files processed: {files_processed}\n")
        f.write(f"# Total unique patterns: {len(sorted_patterns)}\n\n")

        # Write category statistics
        f.write("# Pattern categories:\n")
        for category, count in category_counts.most_common():
            f.write(f"# {category}: {count:,}\n")
        f.write("\n")

        f.write("# Format: pattern<tab>frequency<tab>example_contexts\n\n")

        for pattern, count in sorted_patterns:
            contexts = pattern_contexts.get(pattern, [])
            context_str = " | ".join(contexts)  # Show all contexts (up to 3)
            f.write(f"{pattern}\t{count}\t{context_str}\n")

    # Display summary
    console.print(f"\n[green]Processed {files_processed:,} files")
    console.print(f"Found {len(sorted_patterns):,} unique patterns")
    console.print(f"Results saved to: [blue]{output_file}[/blue]")

    # Show category statistics in a table
    cat_table = Table(title="Pattern Categories")
    cat_table.add_column("Category", style="cyan")
    cat_table.add_column("Count", justify="right", style="green")

    for category, count in category_counts.most_common():
        cat_table.add_row(category, f"{count:,}")

    console.print("\n", cat_table)

    # Show most frequent patterns in a table
    pat_table = Table(title="Most Frequent Patterns")
    pat_table.add_column("Pattern", style="cyan")
    pat_table.add_column("Count", justify="right", style="green")
    pat_table.add_column("Example Context", style="yellow")

    for pattern, count in sorted_patterns[:20]:  # Show top 20 patterns
        context = pattern_contexts.get(pattern, [""])[0]
        pat_table.add_row(pattern, f"{count:,}", context)

    console.print("\n", pat_table)
