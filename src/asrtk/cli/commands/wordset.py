# src/asrtk/cli/commands/wordset.py
"""Command for creating word frequency statistics from input files."""
from pathlib import Path
import re # Import the regular expression module
import rich_click as click
from collections import Counter
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from ...core.text import turkish_lower, sanitize
import webvtt
import traceback

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
        console.log(f"[yellow]Warning: Could not parse VTT file {file_path}. Skipping. Error: {e}[/yellow]")
        return ""

def read_text_content(file_path: Path) -> str:
    """Read content from text file, trying multiple encodings."""
    encodings_to_try = ['utf-8', 'latin-1', 'cp1254']
    content = None
    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
             console.log(f"[red]Error reading file {file_path} with encoding {encoding}: {e}[/red]")
             raise

    if content is None:
        raise ValueError(f"Could not decode file {file_path} with tried encodings: {encodings_to_try}")

    return content

def get_files_recursive(directory: Path, extensions: list[str]) -> list[Path]:
    """Recursively get all files matching specified extensions from directory."""
    found_files = []
    for ext in extensions:
        clean_ext = ext if ext.startswith('.') else f'.{ext}'
        pattern = f'*{clean_ext}'
        found_files.extend(directory.rglob(pattern))
    return list(set(found_files))

def split_words(text: str) -> list[str]:
    """
    Splits text into words using regex.

    Finds sequences of Unicode letters, digits, or underscores.
    Effectively separates words from punctuation.
    """
    # \b ensures we match whole words
    # \w+ matches one or more word characters (letters, numbers, underscore in Unicode)
    words = re.findall(r"\b\w+\b", text)
    return words

@click.command('create-wordset')
@click.argument('input_paths', nargs=-1, type=click.Path(exists=True, readable=True))
@click.option('--extensions', '-e', 'allowed_extensions_input',
              multiple=True,
              default=['txt'],
              help='Allowed file extensions (e.g., -e txt -e csv -e vtt). Multiple allowed. VTT processed specially.')
@click.option('--output', '-o',
              type=click.Path(writable=True, dir_okay=False),
              default="wordset.txt",
              help="Output text file name")
@click.option('--min-freq', '-m',
              type=int,
              default=1,
              help='Minimum frequency threshold for reporting words')
@click.option('--ignore-case/--no-ignore-case',
              default=False, # Changed default to True as it's common for frequency lists
              help='Ignore case when counting words (uses Turkish-aware case folding)')
@click.option('--romanize/--no-romanize',
              default=False,
              help='Romanize Turkish text')
def create_wordset(input_paths, allowed_extensions_input, output, min_freq, ignore_case, romanize):
    """Create word frequency statistics from input files."""
    # ... (rest of the initial setup: input path check, extension processing) ...
    if not input_paths:
        console.print("[yellow]No input paths provided. Nothing to do.[/yellow]")
        return

    allowed_extensions = set()
    for ext_input in allowed_extensions_input:
        parts = ext_input.split(',')
        for part in parts:
            cleaned_ext = part.lower().strip().lstrip('.')
            if cleaned_ext:
                allowed_extensions.add(f'.{cleaned_ext}')

    if not allowed_extensions:
        console.print("[red]Error: No valid file extensions specified or parsed.[/red]", err=True)
        return

    console.print(f"[info]Allowed extensions: {', '.join(sorted(list(allowed_extensions)))}[/info]")
    process_vtt = VTT_EXTENSION in allowed_extensions
    text_extensions = allowed_extensions - {VTT_EXTENSION}

    word_counts = Counter()
    total_words = 0
    processed_files_count = 0
    error_files_count = 0
    skipped_files_count = 0
    all_files_to_process = []

    # --- 1. Collect all files ---
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        # ... (file scanning logic remains the same) ...
        scan_task = progress.add_task("Scanning for files...", total=len(input_paths))

        for input_path_str in input_paths:
            path = Path(input_path_str)
            if path.is_dir():
                try:
                    files_in_dir = get_files_recursive(path, list(allowed_extensions))
                    all_files_to_process.extend(files_in_dir)
                except Exception as e:
                    console.print(f"[red]Error scanning directory {path}: {e}[/red]")
            elif path.is_file():
                file_ext = path.suffix.lower()
                if file_ext in allowed_extensions:
                    all_files_to_process.append(path)
                else:
                    progress.log(f"[yellow]Skipping {path}: Extension '{path.suffix}' not in allowed list.[/yellow]")
                    skipped_files_count += 1
            else:
                 progress.log(f"[yellow]Skipping {path}: Not a valid file or directory.[/yellow]")
                 skipped_files_count += 1
            progress.advance(scan_task)

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
                    progress.log(f"[yellow]Skipping {file_path}: Extension '{file_ext}' not expected.[/yellow]")
                    skipped_files_count += 1
                    progress.advance(process_task)
                    continue

                if not content:
                    progress.log(f"[dim]Skipping {file_path}: No content extracted.[/dim]")
                    progress.advance(process_task)
                    continue

                # --- Text Processing Pipeline ---
                # 1. Basic Cleaning (if sanitize does this)
                content = sanitize(content)

                # 2. Optional: Case Folding (Turkish-aware)
                if ignore_case:
                    content = turkish_lower(content)

                # 3. Optional: Romanization
                if romanize:
                    try:
                        from ...utils import romanize_turkish
                        content = romanize_turkish(content)
                    except ImportError:
                         console.print("[red]Error: --romanize option requires 'unidecode'. Install it or ensure utils are available.[/red]")
                         romanize = False # Disable for subsequent files

                # 4. Tokenize / Split into words using the improved method
                words = split_words(content)
                # --- End Text Processing Pipeline ---


                # Update word counts only if words were found
                if words:
                    word_counts.update(words)
                    total_words += len(words)
                    processed_files_count += 1
                else:
                    # File processed but resulted in no words after splitting
                    processed_files_count += 1 # Still count as processed
                    progress.log(f"[dim]File {file_path} resulted in no words after processing.[/dim]")


            except Exception as e:
                error_files_count += 1
                tb_str = traceback.format_exc()
                progress.log(f"[red]Error processing {file_path}: {e}\nTraceback:\n{tb_str}[/red]")

            finally:
                progress.advance(process_task)

    # --- 3. Report Results ---
    # ... (rest of the reporting logic: sorting, writing to file, printing table/summary) ...
    # ... This part remains largely the same, but the quality of `word_counts` will be much better ...

    console.print("\n[bold]Processing Complete.[/bold]")

    sorted_words = [(word, count) for word, count in word_counts.items()
                   if count >= min_freq and word] # Ensure word is not empty
    sorted_words.sort(key=lambda x: (-x[1], x[0]))

    unique_words_after_min_freq = len(sorted_words)

    output_path = Path(output)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header (consider adding info about the splitter used)
            f.write(f"# Word frequency list\n")
            f.write(f"# Input paths: {', '.join(map(str, input_paths))}\n")
            f.write(f"# Allowed extensions: {', '.join(sorted(list(allowed_extensions)))}\n")
            f.write(f"# Word splitting: regex r'\\b\\w+\\b'\n") # Added info
            f.write(f"# Files found: {total_files_found}\n")
            f.write(f"# Files processed successfully: {processed_files_count}\n")
            if skipped_files_count > 0:
                f.write(f"# Files skipped (wrong extension/type): {skipped_files_count}\n")
            if error_files_count > 0:
                 f.write(f"# Files with errors: {error_files_count}\n")
            f.write(f"# Total words extracted: {total_words:,}\n") # Renamed for clarity
            f.write(f"# Unique words (raw): {len(word_counts):,}\n")
            f.write(f"# Unique words (freq >= {min_freq}): {unique_words_after_min_freq:,}\n")
            f.write(f"# Case sensitive: {not ignore_case}\n")
            f.write(f"# Romanized: {romanize}\n\n")
            f.write("# Format: word<tab>frequency<tab>percentage\n\n")

            if total_words > 0:
                for word, count in sorted_words:
                    percentage = (count / total_words) * 100
                    f.write(f"{word}\t{count}\t{percentage:.4f}%\n")
            else:
                 f.write("# No words found to calculate frequencies.\n")
        console.print(f"[green]Wordset saved successfully to {output_path}[/green]")
    except Exception as e:
        console.print(f"[red]Error writing output file {output_path}: {e}[/red]")

    # Create results table for top words
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

    # Print summary
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
    summary.add_row("Total words extracted:", f"{total_words:,}") # Renamed
    summary.add_row("Unique words (raw):", f"{len(word_counts):,}")
    summary.add_row(f"Unique words (freq ≥ {min_freq}):", f"{unique_words_after_min_freq:,}")
    summary.add_row("Output file:", str(output_path.resolve()))

    console.print("\n[bold underline]Summary:[/bold underline]")
    console.print(summary)