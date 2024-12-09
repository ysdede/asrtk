"""Command for splitting large audio files into smaller chunks with synchronized subtitles."""
from pathlib import Path
import rich_click as click
from pydub import AudioSegment
import webvtt
import subprocess
import tempfile
import os
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.console import Console
from rich.panel import Panel
from ...core.text import format_time

FOUR_HOURS_MS = 4 * 60 * 60 * 1000  # 4 hours in milliseconds
console = Console()

def preprocess_audio(audio_file):
    """Preprocess audio using FFmpeg to resample and convert to mono."""
    with console.status(f"[bold blue]Loading audio file: {audio_file.name}...", spinner="dots"):
        try:
            # Try loading with pydub first
            return AudioSegment.from_file(audio_file), None
        except Exception as e:
            console.print(f"[yellow]Pydub loading failed: {e}")
            console.print("[yellow]Trying FFmpeg preprocessing...")

            # Create temp file for processed audio
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, f"processed_{Path(audio_file).name}")

            try:
                # FFmpeg command to convert to mono and resample
                cmd = [
                    'ffmpeg', '-y',
                    '-i', str(audio_file),
                    '-ac', '1',
                    '-ar', '16000',
                    '-acodec', 'pcm_s16le',
                    temp_file
                ]

                # Run FFmpeg
                subprocess.run(cmd, check=True, capture_output=True)

                # Try loading the processed file
                return AudioSegment.from_file(temp_file), temp_file

            except Exception as ffmpeg_error:
                console.print(f"[red]FFmpeg preprocessing failed: {ffmpeg_error}")
                raise

def find_nearest_split_point(captions, target_time_ms):
    """Find the nearest caption boundary to target time."""
    best_time = 0
    min_diff = float('inf')

    for caption in captions:
        end_time_ms = caption.end_in_seconds * 1000
        diff = abs(end_time_ms - target_time_ms)

        if diff < min_diff:
            min_diff = diff
            best_time = end_time_ms

    return best_time

def split_audio_and_subtitles(audio_file, vtt_file, output_dir, chunk_duration_ms=FOUR_HOURS_MS):
    """Split audio file and corresponding VTT file into chunks."""
    try:
        # Load audio and show file info
        audio, temp_file = preprocess_audio(audio_file)
        total_duration_hours = len(audio) / (1000 * 60 * 60)

        console.print(Panel(f"""
[bold cyan]Audio File Information[/]
File: {audio_file.name}
Duration: {total_duration_hours:.2f} hours
Sample Rate: {audio.frame_rate} Hz
Channels: {audio.channels}
        """))

        captions = list(webvtt.read(vtt_file))
        if not captions:
            console.print(f"[red]No captions found in {vtt_file}")
            return

        # Calculate chunks
        num_chunks = (len(audio) + chunk_duration_ms - 1) // chunk_duration_ms
        console.print(f"[green]Splitting into {num_chunks} chunks...")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            chunk_task = progress.add_task("Processing chunks", total=num_chunks)

            for chunk_num in range(num_chunks):
                progress.update(chunk_task, description=f"Processing chunk {chunk_num + 1}/{num_chunks}")

                chunk_start = chunk_num * chunk_duration_ms
                chunk_end = min(chunk_start + chunk_duration_ms, len(audio))

                # Find nearest caption boundaries
                if chunk_num == 0:
                    actual_start = 0
                else:
                    actual_start = find_nearest_split_point(captions, chunk_start)

                if chunk_num == num_chunks - 1:
                    actual_end = len(audio)
                else:
                    actual_end = find_nearest_split_point(captions, chunk_end)

                # Create chunk paths
                chunk_suffix = f"_chunk{chunk_num+1:02d}"
                audio_out_path = output_dir / f"{audio_file.stem}{chunk_suffix}{audio_file.suffix}"
                vtt_out_path = output_dir / f"{audio_file.stem}{chunk_suffix}.vtt"

                # Process audio chunk
                audio_chunk = audio[actual_start:actual_end]
                audio_chunk.export(str(audio_out_path), format=audio_file.suffix.lstrip('.'))

                # Process captions
                chunk_captions = []
                for caption in captions:
                    start_ms = caption.start_in_seconds * 1000
                    end_ms = caption.end_in_seconds * 1000

                    if start_ms >= actual_start and end_ms <= actual_end:
                        caption.start = format_time(start_ms - actual_start)
                        caption.end = format_time(end_ms - actual_start)
                        chunk_captions.append(caption)

                # Write VTT
                with open(vtt_out_path, 'w', encoding='utf-8') as f:
                    f.write('WEBVTT\n\n')
                    for caption in chunk_captions:
                        f.write(f"{caption.start} --> {caption.end}\n")
                        f.write(f"{caption.text}\n\n")

                chunk_duration = (actual_end - actual_start) / (1000 * 60 * 60)
                progress.console.print(f"[blue]Chunk {chunk_num + 1} completed: {chunk_duration:.2f} hours")
                progress.advance(chunk_task)

    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                console.print("[green]Cleaned up temporary file")
            except Exception as e:
                console.print(f"[yellow]Failed to clean up temporary file: {e}")

@click.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
@click.option("--audio_type", default="m4a", help="Audio file type, default is 'm4a'.")
@click.option("--chunk_hours", default=4.0, type=float, help="Target chunk size in hours.")
def chunk(input_dir: str, output_dir: str, audio_type: str, chunk_hours: float):
    """Split large audio files into smaller chunks with synchronized subtitles."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = list(input_dir.glob(f"*.{audio_type}"))
    if not audio_files:
        console.print(f"[red]No {audio_type} files found in {input_dir}")
        return

    console.print(f"[green]Found {len(audio_files)} audio files to process")

    for audio_file in audio_files:
        vtt_file = audio_file.with_suffix('.tr.vtt')
        if not vtt_file.exists():
            vtt_file = audio_file.with_suffix('.vtt')
            if not vtt_file.exists():
                console.print(f"[yellow]No matching VTT file found for {audio_file}")
                continue

        console.rule(f"[bold cyan]Processing {audio_file.name}")
        split_audio_and_subtitles(
            audio_file,
            vtt_file,
            output_dir,
            chunk_duration_ms=int(chunk_hours * 60 * 60 * 1000)
        )
