"""Command for converting audio files to opus format."""
from pathlib import Path
import rich_click as click
from rich.console import Console
import subprocess
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

console = Console()

def compression_level() -> int:
    """Get a random compression level between 1 and 9."""
    return random.randint(1, 9)

def convert_to_opus(input_file: Path, remove_original: bool = False) -> bool:
    """Convert a single audio file to opus format.

    Args:
        input_file: Path to input audio file
        remove_original: Whether to remove the original file after conversion

    Returns:
        bool: Whether conversion was successful
    """
    output_file = input_file.with_suffix('.opus')

    if output_file.exists():
        console.print(f"[yellow]Skipping {input_file.name} - opus file already exists[/yellow]")
        return False

    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_file),
            "-c:a", "libopus",
            "-b:a", "24K",
            "-application", "voip",
            "-frame_duration", "40",
            "-compression_level", str(compression_level()),
            str(output_file)
        ]

        subprocess.run(cmd, check=True, capture_output=True)

        if remove_original:
            input_file.unlink()

        return True

    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error converting {input_file.name}: {e.stderr.decode()}[/red]")
        return False
    except Exception as e:
        console.print(f"[red]Error processing {input_file.name}: {e}[/red]")
        return False

@click.command('convert-opus')
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--remove-original', is_flag=True, help="Remove original files after successful conversion")
@click.option('--workers', '-w', type=int, default=4, help="Number of worker threads")
@click.option('--input-type', '-t', type=str, default="flac", help="Input audio file type (default: flac)")
def convert_opus(input_dir: str, remove_original: bool, workers: int, input_type: str) -> None:
    """Convert audio files to Opus format.

    This command recursively finds all audio files of specified type in the input directory
    and converts them to Opus format using ffmpeg. The converted files are saved in the same
    location with .opus extension.

    Example:
        # Convert all FLAC files (default), keeping originals
        asrtk convert-opus ./audio_files

        # Convert M4A files and remove originals
        asrtk convert-opus ./audio_files -t m4a --remove-original

        # Use 8 worker threads
        asrtk convert-opus ./audio_files -w 8
    """
    input_path = Path(input_dir)

    # Find all audio files of specified type
    pattern = f"*.{input_type.lower()}"
    audio_files = list(input_path.rglob(pattern))
    if not audio_files:
        console.print(f"No {input_type} files found in input directory")
        return

    console.print(f"Found {len(audio_files)} {input_type} files")

    # Process files
    start_time = time.time()
    completed = 0
    converted = 0

    with console.status("[bold green]Converting files...") as status:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_file = {
                executor.submit(convert_to_opus, f, remove_original): f
                for f in audio_files
            }

            for future in as_completed(future_to_file):
                input_file = future_to_file[future]
                try:
                    if future.result():
                        converted += 1
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

    # Show summary
    console.print(f"\n[green]Completed {completed} files")
    console.print(f"Successfully converted {converted} files to opus")
    if remove_original:
        console.print(f"Removed {converted} original files")
