"""Command for converting audio files between formats."""
from pathlib import Path
import rich_click as click
from rich.console import Console
import time
import signal
from typing import Set
from threading import Event
from concurrent.futures import ThreadPoolExecutor, as_completed
from asrtk.utils.audio_utils import (
    convert_audio,
    check_audio_file_valid,
    probe_audio_file,
    get_audio_duration
)
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

console = Console()
shutdown_event = Event()
active_conversions: Set[Path] = set()

def signal_handler(signum, frame):
    """Handle interrupt signal."""
    console.print("\n[yellow]Shutdown requested. Waiting for current conversions to complete...[/yellow]")
    shutdown_event.set()

def register_shutdown_handler():
    """Register the signal handler for graceful shutdown."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

@click.command('convert')
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--input-type', '-i', type=str, default="mp3", help="Input audio file type (default: mp3)")
@click.option('--output-type', '-o', type=str, default="wav", help="Output audio file type (default: wav)")
@click.option('--remove-original', is_flag=True, help="Remove original files after successful conversion")
@click.option('--workers', '-w', type=int, default=4, help="Number of worker threads")
@click.option('--bitrate', '-b', type=str, default="48k", help="Output bitrate for lossy formats (default: 48k)")
@click.option('--sample-rate', '-sr', type=int, default=16000, help="Output sample rate in Hz (default: 16000)")
def convert(input_dir: str, input_type: str, output_type: str, remove_original: bool, workers: int,
           bitrate: str, sample_rate: int) -> None:
    """Convert audio files between different formats.

    This command recursively finds all audio files of the specified input type in the input directory
    and converts them to the specified output format using ffmpeg.

    Example:
        # Convert MP3 files to WAV format
        asrtk convert ./audio_files -i mp3 -o wav

        # Convert FLAC to MP3 with custom bitrate and sample rate
        asrtk convert ./audio_files -i flac -o mp3 -b 320k -sr 48000

        # Convert WAV to OPUS and remove original files
        asrtk convert ./audio_files -i wav -o opus --remove-original
    """
    register_shutdown_handler()
    shutdown_event.clear()

    input_path = Path(input_dir)

    # Find all audio files of specified input type
    pattern = f"*.{input_type.lower()}"
    audio_files = list(input_path.rglob(pattern))
    if not audio_files:
        console.print(f"No {input_type} files found in input directory")
        return

    console.print(f"Found {len(audio_files)} {input_type} files")
    console.print(f"Converting to {output_type} format")

    # Process files
    start_time = time.time()
    completed = 0
    converted = 0
    broken_files = []

    def convert_file(input_file: Path) -> tuple[bool, str | None]:
        """Convert a single audio file to the target format."""
        output_file = input_file.with_suffix(f'.{output_type}')

        if output_file.exists():
            if check_audio_file_valid(output_file):
                console.print(f"[yellow]Skipping {input_file.name} - output file already exists[/yellow]")
                return False, None
            else:
                console.print(f"[yellow]Found invalid output file for {input_file.name} - reconverting[/yellow]")
                output_file.unlink()

        try:
            active_conversions.add(input_file)

            success, error_msg = convert_audio(
                input_file=input_file,
                output_file=output_file,
                sample_rate=sample_rate,
                output_type=output_type,
                bitrate=bitrate
            )

            if success and remove_original:
                input_file.unlink()

            return success, error_msg

        except Exception as e:
            if output_file.exists():
                output_file.unlink()
            return False, f"Processing error: {str(e)}"
        finally:
            active_conversions.remove(input_file)

    with console.status("[bold green]Converting files...") as status:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_file = {
                executor.submit(convert_file, f): f
                for f in audio_files
            }

            try:
                for future in as_completed(future_to_file):
                    if shutdown_event.is_set():
                        for f in future_to_file:
                            f.cancel()
                        break

                    input_file = future_to_file[future]
                    try:
                        success, error_msg = future.result()
                        if success:
                            converted += 1
                        elif error_msg:
                            broken_files.append((input_file, error_msg))
                        completed += 1

                        # Update progress
                        elapsed = time.time() - start_time
                        files_left = len(audio_files) - completed
                        if completed > 0:
                            avg_time = elapsed / completed
                            est_remaining = (files_left * avg_time) / 60

                            status.update(
                                f"[bold green]Converting files... "
                                f"{completed}/{len(audio_files)} "
                                f"({converted} converted) "
                                f"[yellow]~{est_remaining:.1f}min remaining"
                            )

                    except Exception as e:
                        console.print(f"[red]Error with {input_file}: {e}[/red]")
                        broken_files.append((input_file, str(e)))

            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted by user. Cleaning up...[/yellow]")
                shutdown_event.set()
                executor.shutdown(wait=True)

    # Show final status
    if shutdown_event.is_set():
        console.print("\n[yellow]Conversion stopped by user[/yellow]")
        if active_conversions:
            console.print("[yellow]The following files were being processed when stopped:[/yellow]")
            for file in active_conversions:
                console.print(f"- {file.name}")

    # Show summary
    console.print(f"\n[green]Completed {completed} files")
    console.print(f"Successfully converted {converted} files to {output_type}")
    if remove_original:
        console.print(f"Removed {converted} original files")

    # Report broken files
    if broken_files:
        console.print("\n[red]The following files could not be converted:[/red]")
        for file, error in broken_files:
            console.print(f"[red]- {file.name}: {error}[/red]")

    valid_files = len(audio_files) - len(broken_files) - len(short_files)
    if valid_files > 0:
        console.print(f"\n[green]Statistics for {valid_files} valid files:[/green]")
        console.print(f"[green]- Number of files: {valid_files}[/green]")
        # Convert to hours if more than 1 hour
        if total_duration > 3600:
            console.print(f"[green]- Total duration: {total_duration/3600:.2f} hours[/green]")
        else:
            console.print(f"[green]- Total duration: {total_duration/60:.2f} minutes[/green]")
        console.print(f"[green]- Average duration: {total_duration/valid_files:.2f} seconds[/green]")
        console.print(f"[green]- Shortest file: {min_file_duration:.2f} seconds[/green]")
        console.print(f"[green]- Longest file: {max_duration:.2f} seconds[/green]")
    else:
        console.print("\n[red]No valid files found![/red]")

@click.command('probe-audio')
@click.argument('input_paths', nargs=-1, type=click.Path(exists=True))
@click.option('--min-duration', '-m', type=float, default=0.1,
              help='Minimum duration threshold in seconds')
@click.option('--max-duration', '-M', type=float, default=30.0,
              help='Maximum duration threshold in seconds')
@click.option('--workers', '-w', type=int, default=4,
              help="Number of worker threads")
def probe_audio(input_paths, min_duration, max_duration, workers):
    """Probe audio files for duration and format information."""
    if not input_paths:
        click.echo("No input paths provided", err=True)
        return

    console = Console()
    total_duration = 0
    file_count = 0
    short_clips = []  # Keep track of short clips
    long_clips = []   # Keep track of long clips

    # Create progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        # Process each input path
        for input_path in input_paths:
            path = Path(input_path)

            # Handle directory
            if path.is_dir():
                audio_files = [f for f in path.rglob('*') if f.suffix.lower() in ['.wav', '.mp3', '.flac']]
            else:
                audio_files = [path]

            # Add task for processing files
            task = progress.add_task(f"Processing {path}", total=len(audio_files))

            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = []

                for audio_file in audio_files:
                    futures.append(executor.submit(get_audio_duration, str(audio_file)))

                for audio_file, future in zip(audio_files, futures):
                    try:
                        duration = future.result()
                        if duration < min_duration:
                            short_clips.append((audio_file, duration))
                        elif duration > max_duration:
                            long_clips.append((audio_file, duration))
                        total_duration += duration
                        file_count += 1
                    except Exception as e:
                        progress.log(f"Error processing {audio_file}: {str(e)}")
                    finally:
                        progress.advance(task)

    # Create and display results table
    table = Table(title="Audio Files Analysis")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    table.add_row(
        "Total Duration",
        f"{total_duration:.2f} seconds ({total_duration/3600:.2f} hours)"
    )
    table.add_row("Total Files", str(file_count))
    if file_count > 0:
        table.add_row(
            "Average Duration",
            f"{(total_duration/file_count):.2f} seconds"
        )

    console.print("\n")  # Add spacing
    console.print(table)

    # Report short clips if any
    if short_clips:
        console.print(f"\n[yellow]Short Clips (< {min_duration:.1f}s):[/yellow]")
        short_table = Table(show_header=True)
        short_table.add_column("File", style="cyan")
        short_table.add_column("Duration", justify="right", style="red")

        for file_path, duration in sorted(short_clips, key=lambda x: x[1]):
            short_table.add_row(str(file_path), f"{duration:.2f}s")

        console.print(short_table)

    # Report long clips if any
    if long_clips:
        console.print(f"\n[yellow]Long Clips (> {max_duration:.1f}s):[/yellow]")
        long_table = Table(show_header=True)
        long_table.add_column("File", style="cyan")
        long_table.add_column("Duration", justify="right", style="red")

        for file_path, duration in sorted(long_clips, key=lambda x: x[1], reverse=True):
            long_table.add_row(str(file_path), f"{duration:.2f}s")

        console.print(long_table)

@click.command('probe-mp3')
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--workers', '-w', type=int, default=4, help="Number of worker threads")
@click.option('--min-duration', '-d', type=float, default=1.0,
              help="Minimum duration in seconds (default: 1.0)")
def probe_mp3(input_dir: str, workers: int, min_duration: float) -> None:
    """Probe MP3 files for validity and total duration.

    This command recursively finds all MP3 files in the input directory
    and checks if they are valid using ffprobe. It will report any broken
    or corrupted files and files shorter than minimum duration.

    Example:
        # Probe all MP3 files
        asrtk probe-mp3 ./audio_files

        # Use minimum 2 seconds duration
        asrtk probe-mp3 ./audio_files -d 2.0

        # Use 8 worker threads
        asrtk probe-mp3 ./audio_files -w 8
    """
    # Call probe_audio with MP3 format
    probe_audio(input_dir=input_dir,
               workers=workers,
               format='mp3',
               min_duration=min_duration)
