"""Command for splitting audio files based on VTT subtitles."""
from pathlib import Path
import re
import rich_click as click
from typing import Optional

from ...subs import split_audio_with_subtitles
from ...core.text import natural_sort_key
from ...core.correction import TextCorrector

def extract_youtube_id(filename: str) -> str:
    """Extract YouTube ID from filename containing [YOUTUBE_ID].

    Args:
        filename: Filename potentially containing [YOUTUBE_ID]

    Returns:
        YouTube ID if found, otherwise original filename without extension
    """
    # Look for [SOMETHING] pattern
    match = re.search(r'\[(.*?)\]', filename)
    if match:
        return match.group(1)
    # Fall back to filename without extension
    return Path(filename).stem

def get_matching_pairs(input_dir: Path, audio_type: str, use_filename_fallback: bool = False) -> list[tuple[str, str]]:
    """Get matching audio and VTT file pairs based on YouTube IDs or filenames.

    Args:
        input_dir: Directory containing audio and VTT files
        audio_type: Audio file extension to look for
        use_filename_fallback: Whether to try matching by filename if no YouTube ID pairs found

    Returns:
        List of tuples containing matching (audio_file, vtt_file) pairs
    """
    # Create dictionaries mapping YouTube IDs to files
    vtt_files = {extract_youtube_id(f.name): f.name
                 for f in input_dir.glob("*.vtt")}
    audio_files = {extract_youtube_id(f.name): f.name
                  for f in input_dir.glob(f"*.{audio_type}")}

    # Find matching pairs by YouTube ID
    pairs = []
    for youtube_id in sorted(set(vtt_files.keys()) & set(audio_files.keys())):
        pairs.append((audio_files[youtube_id], vtt_files[youtube_id]))

    # If no pairs found and fallback enabled, try matching by filename
    if not pairs and use_filename_fallback:
        # Get all VTT and audio files
        vtt_files = list(input_dir.glob("*.vtt"))
        audio_files = list(input_dir.glob(f"*.{audio_type}"))

        # Create a mapping of cleaned audio filenames to actual filenames
        audio_map = {Path(f.name).stem: f.name for f in audio_files}

        for vtt_file in vtt_files:
            vtt_stem = Path(vtt_file.name).stem
            # Remove .tr suffix if present
            base_name = vtt_stem.removesuffix('.tr')

            # Check if there's a matching audio file
            if base_name in audio_map:
                pairs.append((audio_map[base_name], vtt_file.name))

    return sorted(pairs, key=lambda x: natural_sort_key(x[0]))

@click.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path(exists=False))
@click.option("--audio_type", default="m4a", help="Audio file type, default is 'm4a'.")
@click.option("--format", "-f", default="wav", help="Output audio format (default: wav)")
@click.option("--pt", default=10, type=int, help="Optional parameter, default is 10.")
@click.option("--tolerance", default=250.0, type=float, help="Tolerance value, default is 250.0.")
@click.option("--forced-alignment", default=True, type=bool, help="Force alignment, default is True.")
@click.option("-fm", "--force-merge", is_flag=True, help="Force merge, default is False.")
@click.option("--keep-effects", is_flag=True, help="Keep effect lines (enclosed in []) instead of skipping them.")
@click.option("--restore-punctuation", is_flag=True, help="Restore punctuation using BERT model.")
@click.option("--use-new-models", is_flag=True, help="Use new BERT models for text correction")
@click.option("--bert-on-cpu", is_flag=True, help="Run BERT models on CPU instead of GPU")
def split(input_dir: str,
         output_dir: str,
         audio_type: str,
         format: str,
         pt: int,
         tolerance: float,
         forced_alignment: bool,
         force_merge: bool = False,
         keep_effects: bool = False,
         restore_punctuation: bool = False,
         use_new_models: bool = False,
         bert_on_cpu: bool = False) -> None:
    """Command to split audio files based on VTT subtitles.

    Examples:
        # Basic usage with default wav output
        asrtk split input_dir/ output_dir/

        # Use MP3 output format
        asrtk split input_dir/ output_dir/ -f mp3

        # Use FLAC output format
        asrtk split input_dir/ output_dir/ -f flac
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    inner_folder_name = input_dir.name

    # Check if the output_dir does not already end with the inner folder name
    if not output_dir.name == inner_folder_name:
        # Append the inner folder name to the output_dir
        output_dir = output_dir / inner_folder_name

    output_dir.mkdir(parents=True, exist_ok=True)

    # First try with YouTube ID matching
    pairs = get_matching_pairs(input_dir, audio_type)

    # If no pairs found, ask user about filename matching
    if not pairs:
        click.echo("No matching audio/VTT pairs found using YouTube IDs.")
        if click.confirm("Would you like to try matching files by filename instead?", default=True):
            pairs = get_matching_pairs(input_dir, audio_type, use_filename_fallback=True)
            if pairs:
                click.echo(f"Found {len(pairs)} matching pairs by filename matching.")
            else:
                click.echo("Still no matching pairs found. Please check your files.")
                return

    if not pairs:
        click.echo("No matching audio/VTT pairs found!")
        return

    click.echo(f"Found {len(pairs)} matching audio/VTT pairs")
    click.echo(f"Output directory: {output_dir}")

    # Initialize text corrector
    text_corrector = TextCorrector(
        use_new_models=use_new_models,
        force_cpu=bert_on_cpu
    )

    for audio_file, vtt_file in pairs:
        # Get episode name from audio file (without extension)
        episode_name = Path(audio_file).stem
        click.echo(f"\nProcessing pair: {audio_file} <-> {vtt_file}")
        click.echo(f"Episode name: {episode_name}")

        episode_dir = output_dir / episode_name
        episode_dir.mkdir(exist_ok=True)

        split_audio_with_subtitles(
            str(input_dir / vtt_file),
            str(input_dir / audio_file),
            episode_dir,
            format=format,
            tolerance=tolerance,
            period_threshold=pt,
            force_merge=force_merge,
            forced_alignment=forced_alignment,
            keep_effects=keep_effects,
            restore_punctuation=restore_punctuation,
            text_corrector=text_corrector
        )
