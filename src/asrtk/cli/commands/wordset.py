"""Command for creating word frequency statistics from input files."""
from pathlib import Path
import rich_click as click
from collections import Counter
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from ...core.text import turkish_lower, sanitize
import webvtt

console = Console()

def read_vtt_content(file_path):
    """Read content from VTT file."""
    captions = webvtt.read(str(file_path))
    return " ".join(caption.text for caption in captions)

def read_text_content(file_path):
    """Read content from text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def get_files_recursive(directory: Path, file_type: str) -> list[Path]:
    """Recursively get all files of specified type from directory."""
    extension = '.vtt' if file_type == 'vtt' else '.txt'
    return list(directory.rglob(f'*{extension}'))

@click.command('create-wordset')
@click.argument('input_paths', nargs=-1, type=click.Path(exists=True))
@click.option('--type', '-t', 'file_type',
              type=click.Choice(['vtt', 'text']),
              default='vtt',
              help='Input file type (vtt or text)')
@click.option('--output', '-o',
              type=str,
              default="wordset.txt",
              help="Output text file name")
@click.option('--min-freq', '-m',
              type=int,
              default=1,
              help='Minimum frequency threshold for reporting words')
@click.option('--ignore-case/--no-ignore-case',
              default=False,
              help='Ignore case when counting words (uses Turkish-aware case folding)')
@click.option('--romanize/--no-romanize',
              default=False,
              help='Romanize Turkish text')
def create_wordset(input_paths, file_type, output, min_freq, ignore_case, romanize):
    """Create word frequency statistics from input files.

    Processes all input files recursively and creates a frequency-sorted word list.
    Input paths can be files or directories. Useful for creating pronunciation
    dictionaries or analyzing vocabulary.
    """
    if not input_paths:
        click.echo("No input paths provided", err=True)
        return

    console = Console()
    word_counts = Counter()
    total_words = 0
    processed_files = 0
    error_files = 0
    all_files = []

    # First collect all files
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        scan_task = progress.add_task("Scanning directories...", total=None)

        for input_path in input_paths:
            path = Path(input_path)
            if path.is_dir():
                files = get_files_recursive(path, file_type)
                all_files.extend(files)
            else:
                all_files.append(path)

        progress.update(scan_task, completed=True)

        # Process files with progress bar
        process_task = progress.add_task(
            "Processing files...",
            total=len(all_files)
        )

        # Process each file
        for file_path in all_files:
            try:
                # Skip files with wrong extension when processing individual files
                if file_path.is_file():
                    expected_ext = '.vtt' if file_type == 'vtt' else '.txt'
                    if file_path.suffix.lower() != expected_ext:
                        progress.log(f"Skipping {file_path}: wrong file type")
                        progress.advance(process_task)
                        continue

                # Read file content based on type
                if file_type == 'vtt':
                    content = read_vtt_content(file_path)
                else:
                    content = read_text_content(file_path)

                # Process content
                content = sanitize(content)

                if ignore_case:
                    content = turkish_lower(content)

                if romanize:
                    from ...utils import romanize_turkish
                    content = romanize_turkish(content)

                # Update word counts
                words = content.split()
                word_counts.update(words)
                total_words += len(words)
                processed_files += 1

            except Exception as e:
                error_files += 1
                progress.log(f"Error processing {file_path}: {str(e)}")

            finally:
                progress.advance(process_task)

    # Filter and sort words by frequency
    sorted_words = [(word, count) for word, count in word_counts.items()
                   if count >= min_freq]
    sorted_words.sort(key=lambda x: (-x[1], x[0]))  # Sort by frequency desc, then word asc

    # Save results with frequencies
    output_path = Path(output)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# Word frequency list from {len(all_files)} files\n")
        f.write(f"# Files processed: {processed_files}\n")
        f.write(f"# Total words processed: {total_words:,}\n")
        f.write(f"# Unique words (freq >= {min_freq}): {len(sorted_words):,}\n")
        f.write(f"# Case sensitive: {not ignore_case}\n")
        f.write(f"# Romanized: {romanize}\n\n")
        f.write("# Format: word<tab>frequency<tab>percentage\n\n")

        # Write frequencies and percentages
        for word, count in sorted_words:
            percentage = (count / total_words) * 100
            f.write(f"{word}\t{count}\t{percentage:.4f}%\n")

    # Create results table
    table = Table(
        title="Word Frequency Analysis",
        title_style="bold magenta",
        border_style="blue"
    )
    table.add_column("Word", style="cyan")
    table.add_column("Frequency", justify="right", style="yellow")
    table.add_column("Percentage", justify="right", style="yellow")

    # Show top 20 words in table
    for word, count in sorted_words[:20]:
        percentage = (count / total_words) * 100
        table.add_row(
            word,
            f"{count:,}",
            f"{percentage:.2f}%"
        )

    console.print("\n")  # Add spacing
    console.print(table)

    # Print summary with styling
    summary = Table.grid(padding=1)
    summary.add_column(style="bold")
    summary.add_column()

    summary.add_row("Total words:", f"{total_words:,}")
    summary.add_row("Unique words:", f"{len(sorted_words):,}")
    summary.add_row("Files processed:", str(processed_files))
    if error_files:
        summary.add_row("Files with errors:", f"[red]{error_files}[/red]")
    summary.add_row("Output file:", str(output_path))

    console.print("\n[bold]Summary:[/bold]")
    console.print(summary)
