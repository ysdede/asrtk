"""Command for trimming silence from audio files using Silero VAD."""
from pathlib import Path
import rich_click as click
from rich.console import Console
import torch
import torchaudio
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import time
import gc
import subprocess
from typing import Optional, List, Tuple
from asrtk.utils.audio_utils import (
    probe_audio_file,
    get_high_quality_ffmpeg_args,
    get_format_specific_args
)
from pydub import AudioSegment
import io
from asrtk.subs import _load_silero_model

console = Console()

def process_file(input_file: Path, output_dir: Path,
                model, get_speech_timestamps,
                threshold: float = 0.6,
                bitrate: str = "48k",
                debug: bool = False) -> Tuple[bool, Optional[str]]:
    """Process a single audio file - detect and trim silence."""
    try:
        # Check if output file already exists
        output_file = output_dir / input_file.name
        if output_file.exists():
            try:
                audio_info = probe_audio_file(output_file)
                if debug:
                    console.print(f"[dim]Debug - Probe result for {input_file.name}:[/dim]")
                    console.print(f"[dim]{audio_info}[/dim]")

                if (audio_info.valid and audio_info.duration > 0 and
                    audio_info.sample_rate == 16000 and not audio_info.error):
                    return True, "File already exists and is valid (skipped)"
                else:
                    output_file.unlink()
                    console.print(f"[yellow]Found invalid output file for {input_file.name} - will reprocess[/yellow]")
            except Exception as e:
                output_file.unlink()
                if debug:
                    console.print(f"[yellow]Found invalid output file for {input_file.name} - will reprocess ({str(e)})[/yellow]")
                else:
                    console.print(f"[yellow]Found invalid output file for {input_file.name} - will reprocess[/yellow]")

        # Load and convert audio to 16kHz
        audio = AudioSegment.from_file(str(input_file))

        # Convert to WAV in memory
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format='wav')
        wav_buffer.seek(0)

        # Load into PyTorch and convert to mono if needed
        waveform, orig_sr = torchaudio.load(wav_buffer)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample to 16kHz if needed
        if orig_sr != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sr,
                new_freq=16000,
                # resampling_method="kaiser_window",
                # lowpass_filter_width=512,
                # rolloff=0.95,
                # beta=14.0
            )
            waveform = resampler(waveform)

        # Get speech timestamps using Silero VAD
        with torch.no_grad():
            timestamps = get_speech_timestamps(
                waveform[0],
                model,
                threshold=threshold,
                sampling_rate=16000,
                return_seconds=True
            )

        if not timestamps:
            return False, "No speech detected"

        # Trim audio
        start_frame = int(timestamps[0]['start'] * 16000)
        end_frame = int(timestamps[-1]['end'] * 16000)
        trimmed = waveform[:, start_frame:end_frame]

        # Save trimmed audio
        wav_buffer = io.BytesIO()
        torchaudio.save(wav_buffer, trimmed, 16000, format='wav')
        wav_buffer.seek(0)

        # Convert to target format with high quality settings
        cmd = [
            'ffmpeg', '-y',
            '-hide_banner',
            '-loglevel', 'error',
            '-f', 'wav',
            '-i', 'pipe:',
            '-map_metadata', '-1',
            '-map', '0:a:0',
            '-ac', '1',
            '-ar', '16000'
        ]
        cmd.extend(get_format_specific_args(output_file.suffix.lstrip('.'), bitrate))
        cmd.append(str(output_file))

        subprocess.run(cmd, input=wav_buffer.getvalue(), check=True, capture_output=True)

        return True, f"Trimmed {timestamps[0]['start']:.2f}s -> {timestamps[-1]['end']:.2f}s"

    except Exception as e:
        return False, str(e)

@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--input-type', '-i', type=str, default="mp3", help="Input audio file type (default: mp3)")
@click.option('--threshold', '-t', type=float, default=0.5, help="VAD threshold (0.0-1.0, default: 0.5)")
@click.option('--workers', '-w', type=int, default=4, help="Number of worker threads")
@click.option('--bitrate', '-b', type=str, default="48k", help="Output bitrate for lossy formats (default: 48k)")
@click.option('--debug/--no-debug', default=False, help="Enable debug logging")
def trim(input_dir: str, output_dir: str, input_type: str, threshold: float, workers: int,
         bitrate: str = "48k", debug: bool = False) -> None:
    """Trim silence from audio files using Silero VAD.

    This command processes audio files to remove silence from the beginning and end,
    using Silero VAD (Voice Activity Detection). All audio is converted to 16kHz mono.

    Example:
        # Trim silence from MP3 files
        asrtk trim ./audio_files ./trimmed_files

        # Process WAV files with custom threshold
        asrtk trim ./audio_files ./trimmed_files -i wav -t 0.7

        # Use higher bitrate for MP3 output
        asrtk trim ./audio_files ./trimmed_files -b 192k

        # Use 8 worker threads
        asrtk trim ./audio_files ./trimmed_files -w 8
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Safety check - ensure input and output dirs are different
    if input_path.resolve() == output_path.resolve():
        console.print("[red]Error: Input and output directories must be different to prevent overwriting original files[/red]")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    # Find audio files
    pattern = f"*.{input_type.lower()}"
    audio_files = list(input_path.rglob(pattern))
    if not audio_files:
        console.print(f"No {input_type} files found in input directory")
        return

    console.print(f"Found {len(audio_files)} {input_type} files")
    console.print("Starting processing...")

    # Load Silero VAD model once using the cached loader from subs.py
    model, utils = _load_silero_model()
    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils

    # Process files
    start_time = time.time()
    completed = 0
    processed = 0
    failed_files = []

    with console.status("[bold green]Processing files...") as status:
        for audio_file in audio_files:
            try:
                success, message = process_file(
                    audio_file,
                    output_path,
                    model,
                    get_speech_timestamps,
                    threshold,
                    bitrate,
                    debug
                )

                if success:
                    processed += 1
                    # Only show skipped files in debug mode
                    if "skipped" in message:
                        if debug:
                            console.print(f"[green]✓ {audio_file.name}: {message}[/green]")
                    else:
                        console.print(f"[green]✓ {audio_file.name}: {message}[/green]")
                else:
                    failed_files.append((audio_file, message))
                    console.print(f"[yellow]✗ {audio_file.name}: {message}[/yellow]")

            except Exception as e:
                console.print(f"[red]Error processing {audio_file.name}: {str(e)}[/red]")
                failed_files.append((audio_file, str(e)))

            completed += 1

            # Update progress
            elapsed = time.time() - start_time
            files_left = len(audio_files) - completed
            if completed > 0:
                avg_time = elapsed / completed
                est_remaining = (files_left * avg_time) / 60

                status.update(
                    f"[bold green]Processing files... "
                    f"{completed}/{len(audio_files)} "
                    f"({processed} successfully trimmed) "
                    f"[yellow]~{est_remaining:.1f}min remaining"
                )

    # Show final summary
    console.print(f"\n[green]Completed {completed}/{len(audio_files)} files")
    console.print(f"Successfully processed {processed} files")

    if failed_files:
        console.print("\n[red]The following files could not be processed:[/red]")
        for file, error in failed_files:
            console.print(f"[red]- {file.name}: {error}[/red]")
