"""Command for splitting audio files based on VTT subtitles."""
from pathlib import Path
import re
import rich_click as click

from ...subs import split_audio_with_subtitles
from ...core.text import natural_sort_key

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

def get_matching_pairs(input_dir: Path, audio_type: str) -> list[tuple[str, str]]:
    """Get matching audio and VTT file pairs based on YouTube IDs.

    Args:
        input_dir: Directory containing audio and VTT files
        audio_type: Audio file extension to look for

    Returns:
        List of tuples containing matching (audio_file, vtt_file) pairs
    """
    # Create dictionaries mapping YouTube IDs to files
    vtt_files = {extract_youtube_id(f.name): f.name
                 for f in input_dir.glob("*.vtt")}
    audio_files = {extract_youtube_id(f.name): f.name
                  for f in input_dir.glob(f"*.{audio_type}")}

    # Find matching pairs
    pairs = []
    for youtube_id in sorted(set(vtt_files.keys()) & set(audio_files.keys())):
        pairs.append((audio_files[youtube_id], vtt_files[youtube_id]))

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
def split(input_dir: str,
         output_dir: str,
         audio_type: str,
         format: str,
         pt: int,
         tolerance: float,
         forced_alignment: bool,
         force_merge: bool = False,
         keep_effects: bool = False,
         restore_punctuation: bool = False) -> None:
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

    # Get matching pairs using YouTube IDs
    pairs = get_matching_pairs(input_dir, audio_type)

    if not pairs:
        click.echo("No matching audio/VTT pairs found!")
        return

    click.echo(f"Found {len(pairs)} matching audio/VTT pairs")
    click.echo(f"Output directory: {output_dir}")

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
            restore_punctuation=restore_punctuation
        )
