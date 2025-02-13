import subprocess
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass
from functools import lru_cache

@dataclass
class AudioInfo:
    """Container for audio file information."""
    valid: bool
    sample_rate: Optional[int] = None
    duration: Optional[float] = None
    error: Optional[str] = None

@lru_cache(maxsize=1024)
def probe_audio_file(file_path: Path) -> AudioInfo:
    """Probe audio file for validity and properties in a single ffprobe call.

    Args:
        file_path: Path to audio file

    Returns:
        AudioInfo: Container with audio file properties
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate",
            "-show_entries", "format=duration",
            "-of", "json",
            str(file_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return AudioInfo(valid=False, error=result.stderr)

        data = json.loads(result.stdout)

        # Extract sample rate and duration
        sample_rate = None
        if 'streams' in data and data['streams']:
            sample_rate = int(data['streams'][0].get('sample_rate', 0))

        duration = None
        if 'format' in data:
            duration = float(data['format'].get('duration', 0))

        return AudioInfo(
            valid=True,
            sample_rate=sample_rate,
            duration=duration
        )

    except Exception as e:
        return AudioInfo(valid=False, error=str(e))

def check_audio_file_valid(file_path: Path) -> bool:
    """Check if an audio file is valid."""
    info = probe_audio_file(file_path)
    return info.valid

def get_audio_sample_rate(file_path: Path) -> Optional[int]:
    """Get audio file sample rate."""
    info = probe_audio_file(file_path)
    return info.sample_rate if info.valid else None

def get_audio_duration(file_path: Path) -> Optional[float]:
    """Get audio file duration in seconds."""
    info = probe_audio_file(file_path)
    return info.duration if info.valid else None

def convert_audio(input_file: Path,
                 output_file: Path,
                 sample_rate: Optional[int] = None,
                 output_type: Optional[str] = None,
                 bitrate: str = "48k") -> Tuple[bool, Optional[str]]:
    """Convert audio file with high-quality resampling."""
    try:
        if output_type is None:
            output_type = output_file.suffix.lstrip('.')

        # Get input file info once
        input_info = probe_audio_file(input_file)
        if not input_info.valid:
            return False, f"Invalid input file: {input_info.error}"

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_file),
            "-map_metadata", "-1",  # Remove metadata
            "-map", "0:a:0",  # Select first audio stream
            "-ac", "1"  # Convert to mono
        ]

        # Add resampling args only if needed
        if sample_rate and input_info.sample_rate and input_info.sample_rate != sample_rate:
            cmd.extend(get_high_quality_ffmpeg_args(sample_rate))
            logging.info(f"Resampling from {input_info.sample_rate}Hz to {sample_rate}Hz")
        else:
            logging.info(f"Skipping resampling - input at {input_info.sample_rate}Hz")

        # Add format-specific encoding args
        cmd.extend(get_format_specific_args(output_type, bitrate))

        # Add output file
        cmd.append(str(output_file))

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return False, f"FFmpeg error: {result.stderr}"

        return True, None

    except Exception as e:
        return False, f"Conversion error: {str(e)}"

def get_high_quality_ffmpeg_args(sample_rate: Optional[int] = None) -> list[str]:
    """Get high-quality FFmpeg resampling arguments using SoX resampler.

    Args:
        sample_rate: Target sample rate in Hz. If None, no resampling is performed.

    Returns:
        list[str]: FFmpeg arguments for high-quality resampling
    """
    args = []

    if sample_rate:
        # Use high quality SoX resampler with same settings as in subs.py
        args.extend([
            "-af", "aresample=resampler=soxr:precision=32:dither_method=triangular",
            "-ar", str(sample_rate)
        ])

    return args

def get_format_specific_args(output_type: str, bitrate: str = "48k") -> List[str]:
    """Get format-specific FFmpeg arguments."""
    if output_type.lower() in {'mp3', 'mp2'}:
        return [
            '-c:a', 'libmp3lame',     # Use LAME encoder
            '-q:a', '8',              # VBR quality (0-9, higher number = lower bitrate, 8 ~32-48kbps for mono)
            '-maxrate', '80k',        # Maximum allowed bitrate
            # '-bufsize', '48k',        # Buffer size matching max bitrate
            '-ac', '1',               # Ensure mono output
            '-compression_level', '7', # Maximum compression effort
            '-ar', '16000',           # Set sample rate to 16kHz
            '-map_metadata', '-1'      # Strip metadata to save space
        ]
    elif output_type.lower() == "wav":
        return [
            '-c:a', 'pcm_s16le'  # 16-bit PCM
        ]
    else:
        return []
