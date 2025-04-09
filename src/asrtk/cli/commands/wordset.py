# src/asrtk/cli/commands/wordset.py
"""Command for creating word frequency statistics from input files."""
from pathlib import Path
import rich_click as click
from collections import Counter
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from ...core.text import turkish_lower, sanitize
import webvtt
import traceback # Import traceback for better error logging

console = Console()

# Constants for easier management
VTT_EXTENSION = '.vtt'

def read_vtt_content(file_path: Path) -> str:
    """Read content from VTT file."""
    try:
        captions = webvtt.read(str(file_path))
        # Filter out empty or whitespace-only captions
        return " ".join(caption.text.strip() for caption in captions if caption.text.strip())
    except Exception as e:
        # Improve error reporting for VTT parsing issues
        console.log(f"[yellow]Warning: Could not parse VTT file {file_path}. Skipping. Error: {e}[/yellow]")
        # Optionally re-raise or return empty string depending on desired strictness
        # raise  # Re-raise if parsing errors should stop the process for this file
        return "" # Return empty string to continue processing other files

def read_text_content(file_path: Path) -> str:
    """Read content from text file."""
    # Try common encodings if utf-8 fails
    encodings_to_try = ['utf-8', 'latin-1', 'cp1254'] # Common Turkish/general encodings
    content = None
    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            # If successful, break the loop
            break
        except UnicodeDecodeError:
            continue # Try next encoding
        except Exception as e:
             # Catch other file reading errors
             console.log(f"[red]Error reading file {file_path} with encoding {encoding}: {e}[/red]")
             raise # Re-raise other exceptions

    if content is None:
        # If all encodings failed
        raise ValueError(f"Could not decode file {file_path} with tried encodings: {encodings_to_try}")

    return content

def get_files_recursive(directory: Path, extensions: list[str]) -> list[Path]:
    """Recursively get all files matching specified extensions from directory."""
    found_files = []
    for ext in extensions:
        # Ensure extension starts with a dot
        clean_ext = ext if ext.startswith('.') else f'.{ext}'
        # Use lower() for case-insensitive matching on file systems that might be case-sensitive
        # Note: rglob is generally case-insensitive on Windows/macOS, case-sensitive on Linux.
        # This pattern matching is a simple approach.
        # For truly robust case-insensitivity on Linux, more complex find commands or manual checks might be needed.
        # However, rglob often works well enough in practice.
        pattern = f'*{clean_ext}'
        found_files.extend(directory.rglob(pattern))
        # If case-insensitivity is critical on Linux, consider checking upper/mixed case too:
        # pattern_upper = f'*{clean_ext.upper()}'
        # found_files.extend(directory.rglob(pattern_upper))
        # ...potentially more patterns for mixed case...

    # Return unique paths
    return list(set(found_files))

@click.command('create-wordset')
@click.argument('input_paths', nargs=-1, type=click.Path(exists=True, readable=True))
@click.option('--extensions', '-e', 'allowed_extensions_input',
              multiple=True,
              default=['txt'], # Default to looking for .txt files
              help='Allowed file extensions to process (e.g., -e txt -e csv -e vtt). Specify multiple times or separate by comma. VTT files are processed specially.')
              # Note: Click handles multiple=True well. Comma separation isn't directly supported here but multiple flags work.
@click.option('--output', '-o',
              type=click.Path(writable=True, dir_okay=False), # Ensure output is a writable file path
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
def create_wordset(input_paths, allowed_extensions_input, output, min_freq, ignore_case, romanize):
    """Create word frequency statistics from input files.

    Processes all files matching the specified --extensions recursively from
    input directories, or processes specified input files if they match the
    allowed extensions. Creates a frequency-sorted word list.

    VTT files (.vtt) are parsed specifically to extract caption text. All other
    specified file types are treated as plain text.
    """
    if not input_paths:
        console.print("[yellow]No input paths provided. Nothing to do.[/yellow]")
        return

    # Process and normalize allowed extensions
    allowed_extensions = set()
    for ext_input in allowed_extensions_input:
        # Allow comma-separated extensions within a single -e argument as well
        parts = ext_input.split(',')
        for part in parts:
            cleaned_ext = part.lower().strip().lstrip('.')
            if cleaned_ext: # Avoid empty strings
                allowed_extensions.add(f'.{cleaned_ext}')

    if not allowed_extensions:
        console.print("[red]Error: No valid file extensions specified or parsed.[/red]", err=True)
        return

    console.print(f"[info]Allowed extensions: {', '.join(sorted(list(allowed_extensions)))}[/info]")
    process_vtt = VTT_EXTENSION in allowed_extensions
    text_extensions = allowed_extensions - {VTT_EXTENSION} # All non-VTT allowed extensions

    word_counts = Counter()
    total_words = 0
    processed_files_count = 0
    error_files_count = 0
    skipped_files_count = 0
    all_files_to_process = []

    # --- 1. Collect all files ---
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        scan_task = progress.add_task("Scanning for files...", total=len(input_paths))

        for input_path_str in input_paths:
            path = Path(input_path_str)
            if path.is_dir():
                try:
                    # Find files with any of the allowed extensions recursively
                    files_in_dir = get_files_recursive(path, list(allowed_extensions))
                    all_files_to_process.extend(files_in_dir)
                except Exception as e:
                    console.print(f"[red]Error scanning directory {path}: {e}[/red]")
                    # Optionally skip this directory or stop
            elif path.is_file():
                # Check if the explicitly provided file has an allowed extension
                file_ext = path.suffix.lower()
                if file_ext in allowed_extensions:
                    all_files_to_process.append(path)
                else:
                    progress.log(f"[yellow]Skipping {path}: Extension '{path.suffix}' not in allowed list.[/yellow]")
                    skipped_files_count += 1
            else:
                 # This case should be caught by click's exists=True, but good to handle
                 progress.log(f"[yellow]Skipping {path}: Not a valid file or directory.[/yellow]")
                 skipped_files_count += 1
            progress.advance(scan_task)

        # Remove duplicates that might arise from specifying overlapping paths/files
        unique_files = sorted(list(set(all_files_to_process)))
        total_files_found = len(unique_files)
        console.print(f"[info]Found {total_files_found} potential files to process.[/info]")

    if not unique_files:
        console.print("[yellow]No files matching the specified extensions found in the input paths.[/yellow]")
        return

    # --- 2. Process collected files ---
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total} | ETA: {task.time_remaining}"),
        console=console,
    ) as progress:
        process_task = progress.add_task(
            "Processing files...",
            total=total_files_found
        )

        for file_path in unique_files:
            content = None
            try:
                file_ext = file_path.suffix.lower()

                # Determine how to read the file based on its actual extension
                if process_vtt and file_ext == VTT_EXTENSION:
                    content = read_vtt_content(file_path)
                elif file_ext in text_extensions:
                    content = read_text_content(file_path)
                else:
                    # This should not happen if collection logic is correct, but safeguard
                    progress.log(f"[yellow]Skipping {file_path}: Extension '{file_ext}' was not expected at processing stage.[/yellow]")
                    skipped_files_count += 1
                    progress.advance(process_task)
                    continue # Skip to the next file

                # If content is empty (e.g., VTT parsing failed gracefully, or file is empty), skip processing steps
                if not content:
                    progress.log(f"[dim]Skipping {file_path}: No content extracted.[/dim]")
                    # Don't count as processed, but advance the progress bar
                    progress.advance(process_task)
                    continue


                # Process content (same as before)
                content = sanitize(content) # Basic cleaning

                if ignore_case:
                    content = turkish_lower(content)

                if romanize:
                    # Lazy import if needed, or ensure it's available
                    try:
                        from ...utils import romanize_turkish
                        content = romanize_turkish(content)
                    except ImportError:
                         console.print("[red]Error: --romanize option requires 'unidecode' library. Install it (`pip install unidecode`) or ensure asrtk utils are available.[/red]")
                         # Decide whether to exit or continue without romanization
                         romanize = False # Disable for subsequent files
                         # Or raise click.Abort()

                # Update word counts
                words = content.split()
                if words: # Only update if there are words
                    word_counts.update(words)
                    total_words += len(words)
                    processed_files_count += 1
                else:
                    # File processed but resulted in no words after cleaning
                    processed_files_count += 1 # Count it as processed
                    progress.log(f"[dim]File {file_path} resulted in no words after processing.[/dim]")


            except Exception as e:
                error_files_count += 1
                # Provide more detail in case of unexpected errors
                tb_str = traceback.format_exc()
                progress.log(f"[red]Error processing {file_path}: {e}\nTraceback:\n{tb_str}[/red]")

            finally:
                # Always advance progress bar for each file attempted
                progress.advance(process_task)

    # --- 3. Report Results ---
    console.print("\n[bold]Processing Complete.[/bold]")

    # Filter and sort words by frequency
    sorted_words = [(word, count) for word, count in word_counts.items()
                   if count >= min_freq]
    sorted_words.sort(key=lambda x: (-x[1], x[0]))  # Sort by frequency desc, then word asc

    unique_words_after_min_freq = len(sorted_words)

    # Save results with frequencies
    output_path = Path(output)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Word frequency list\n")
            f.write(f"# Input paths: {', '.join(map(str, input_paths))}\n")
            f.write(f"# Allowed extensions: {', '.join(sorted(list(allowed_extensions)))}\n")
            f.write(f"# Files found: {total_files_found}\n")
            f.write(f"# Files processed successfully: {processed_files_count}\n")
            if skipped_files_count > 0:
                f.write(f"# Files skipped (wrong extension/type): {skipped_files_count}\n")
            if error_files_count > 0:
                 f.write(f"# Files with errors: {error_files_count}\n")
            f.write(f"# Total words processed: {total_words:,}\n")
            f.write(f"# Unique words (raw): {len(word_counts):,}\n")
            f.write(f"# Unique words (freq >= {min_freq}): {unique_words_after_min_freq:,}\n")
            f.write(f"# Case sensitive: {not ignore_case}\n")
            f.write(f"# Romanized: {romanize}\n\n")
            f.write("# Format: word<tab>frequency<tab>percentage\n\n")

            # Write frequencies and percentages
            if total_words > 0: # Avoid division by zero
                for word, count in sorted_words:
                    percentage = (count / total_words) * 100
                    f.write(f"{word}\t{count}\t{percentage:.4f}%\n")
            else:
                 f.write("# No words found to calculate frequencies.\n")
        console.print(f"[green]Wordset saved successfully to {output_path}[/green]")
    except Exception as e:
        console.print(f"[red]Error writing output file {output_path}: {e}[/red]")


    # Create results table for top words (only if words were found)
    if total_words > 0 and sorted_words:
        table = Table(
            title="Top 20 Words",
            title_style="bold magenta",
            border_style="blue",
            show_header=True, header_style="bold blue"
        )
        table.add_column("Word", style="cyan", min_width=15)
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
        console.print("\n")
        console.print(table)
    elif processed_files_count > 0:
        console.print("\n[yellow]No words met the minimum frequency criteria or no words were found after processing.[/yellow]")

    # Print summary with styling
    summary = Table.grid(padding=(0, 1))
    summary.add_column(style="bold blue", justify="right")
    summary.add_column()

    summary.add_row("Input paths:", f"{len(input_paths)}")
    summary.add_row("Allowed extensions:", f"{', '.join(sorted(list(allowed_extensions)))}")
    summary.add_row("Files found:", f"{total_files_found:,}")
    summary.add_row("Files processed:", f"{processed_files_count:,}")
    if skipped_files_count > 0:
         summary.add_row("Files skipped:", f"[yellow]{skipped_files_count:,}[/yellow]")
    if error_files_count > 0:
        summary.add_row("Files with errors:", f"[red]{error_files_count:,}[/red]")
    summary.add_row("Total words:", f"{total_words:,}")
    summary.add_row("Unique words (raw):", f"{len(word_counts):,}")
    summary.add_row(f"Unique words (freq ≥ {min_freq}):", f"{unique_words_after_min_freq:,}")
    summary.add_row("Output file:", str(output_path.resolve())) # Show full path

    console.print("\n[bold underline]Summary:[/bold underline]")
    console.print(summary)
