"""Command for splitting audio files based on VTT subtitles."""
from pathlib import Path
import rich_click as click

from ...subs import split_audio_with_subtitles
from ...core.text import natural_sort_key

@click.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path(exists=False))
@click.option("--audio_type", default="m4a", help="Audio file type, default is 'm4a'.")
@click.option("--format", "-f", default="wav", help="Output audio format (default: wav)")
@click.option("--pt", default=10, type=int, help="Optional parameter, default is 10.")
@click.option("--tolerance", default=250.0, type=float, help="Tolerance value, default is 250.0.")
@click.option("--forced-alignment", default=True, type=bool, help="Force alignment, default is True.")
@click.option("-fm", "--force-merge", is_flag=True, help="Force merge, default is False.")
def split(input_dir: str,
         output_dir: str,
         audio_type: str,
         format: str,
         pt: int,
         tolerance: float,
         forced_alignment: bool,
         force_merge: bool = False) -> None:
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

    audio_file_filter = f'*.{audio_type}'
    # Collect all .vtt and audio files in the directory
    vtt_files = [file.name for file in input_dir.glob("*.vtt")]
    acc_files = [file.name for file in input_dir.glob(audio_file_filter)]

    acc_files = sorted(acc_files, key=natural_sort_key)

    pairs = []
    for acc_file in acc_files:
        for vtt_file in vtt_files:
            if acc_file.split(".")[0] == vtt_file.split(".")[0]:
                pairs.append((acc_file, vtt_file))

    click.echo(f"{output_dir}")

    for p in pairs:
        # split filename with ., drop last element, recompile as string back
        episode_name = ".".join(p[0].split(".")[:-1])
        click.echo(f"{p}, {episode_name}")

        episode_dir = output_dir / episode_name
        episode_dir.mkdir(exist_ok=True)
        click.echo(episode_dir)
        split_audio_with_subtitles(
            f"{input_dir}/{p[1]}",
            f"{input_dir}/{p[0]}",
            episode_dir,
            format=format,
            tolerance=tolerance,
            period_threshold=pt,
            force_merge=force_merge,
            forced_alignment=forced_alignment
        )
