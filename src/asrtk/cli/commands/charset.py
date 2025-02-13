"""Command to analyze character sets in input files."""
import rich_click as click
from pathlib import Path
import webvtt
from collections import Counter
from rich.table import Table
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live

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

@click.command('create-charset')
@click.argument('input_paths', nargs=-1, type=click.Path(exists=True))
@click.option('--type', '-t', 'file_type',
              type=click.Choice(['vtt', 'text']),
              default='vtt',
              help='Input file type (vtt or text)')
@click.option('--min-freq', '-m',
              type=int,
              default=1,
              help='Minimum frequency threshold for reporting characters')
def create_charset(input_paths, file_type, min_freq):
    """Analyze character sets in input files.

    Creates a report of unique characters and their frequencies from the input files.
    Input paths can be files or directories (which will be scanned recursively).
    """
    if not input_paths:
        click.echo("No input paths provided", err=True)
        return

    console = Console()
    total_chars = Counter()
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
                if file_path.is_file():  # Check if it's a file
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

                # Update total character counts
                total_chars.update(content)
                processed_files += 1

            except Exception as e:
                error_files += 1
                progress.log(f"Error processing {file_path}: {str(e)}")

            finally:
                progress.advance(process_task)

    # Create results table
    table = Table(
        title="Character Set Analysis",
        title_style="bold magenta",
        border_style="blue"
    )
    table.add_column("Character", justify="center", style="cyan")
    table.add_column("Unicode", justify="left", style="green")
    table.add_column("Frequency", justify="right", style="yellow")
    table.add_column("Percentage", justify="right", style="yellow")

    total_count = sum(total_chars.values())

    # Add rows for characters meeting minimum frequency
    for char, freq in sorted(total_chars.items(), key=lambda x: (-x[1], x[0])):
        if freq >= min_freq:
            percentage = (freq / total_count) * 100
            unicode_value = f"U+{ord(char):04X}"
            table.add_row(
                char if char.isprintable() else f"<{unicode_value}>",
                unicode_value,
                f"{freq:,}",  # Add thousand separators
                f"{percentage:.2f}%"
            )

    console.print("\n")  # Add some spacing
    console.print(table)

    # Print summary with styling
    unique_chars = len([c for c, f in total_chars.items() if f >= min_freq])

    summary = Table.grid(padding=1)
    summary.add_column(style="bold")
    summary.add_column()

    summary.add_row("Total characters:", f"{total_count:,}")
    summary.add_row("Unique characters:", str(unique_chars))
    summary.add_row("Files processed:", str(processed_files))
    if error_files:
        summary.add_row("Files with errors:", f"[red]{error_files}[/red]")

    console.print("\n[bold]Summary:[/bold]")
    console.print(summary)
