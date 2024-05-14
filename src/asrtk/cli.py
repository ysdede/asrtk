"""Main CLI for asrtk."""
import rich_click as click

from . import __version__

context_settings = {"help_option_names": ["-h", "--help"]}
help_config = click.RichHelpConfiguration(
    width=88,
    show_arguments=True,
    use_rich_markup=True,
)


@click.group(context_settings=context_settings)
@click.rich_config(help_config=help_config)
@click.version_option(__version__, "-v", "--version")
def cli() -> None:
    """An open-source Python toolkit designed to streamline the development and enhancement of ASR systems."""


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path(exists=False))
@click.option("--audio_type", default="m4a", help="Audio file type, default is 'm4a'.")
@click.option("--pt", default=10, type=int, help="Optional parameter, default is 10.")
@click.option("--tolerance", default=250.0, type=float, help="Tolerance value, default is 250.0.")
# @click.option("--force-merge", default=False, type=bool, help="Force merge, default is False.")
@click.option("--forced-alignment", default=True, type=bool, help="Force alignment, default is True.")
@click.option("-fm", "--force-merge", is_flag=True, help="Force merge, default is False.")
def split(input_dir, output_dir, audio_type, pt, tolerance, forced_alignment, force_merge: bool=False):
    """ Command to split audio files. """
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Audio type: {audio_type}")
    print(f"Optional parameter pt: {pt}")
    print(f"Tolerance: {tolerance}")
    print(f"Forced alignment: {forced_alignment}")
    print(f"Forced merge: {force_merge}")

    from pathlib import Path
    from asrtk.subs import split_audio_with_subtitles
    from asrtk.utils import natural_sort_key

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    inner_folder_name = input_dir.name

    # Check if the output_dir does not already end with the inner folder name
    if not output_dir.name == inner_folder_name:
        # Append the inner folder name to the output_dir
        output_dir = output_dir / inner_folder_name
        
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_file_filter = f'*.{audio_type}'
    # Collect all .vtt and .m4a files in the directory
    vtt_files = [file.name for file in input_dir.glob("*.vtt")]
    acc_files = [file.name for file in input_dir.glob(audio_file_filter)]

    acc_files = sorted(acc_files, key=natural_sort_key)

    pairs = []
    for acc_file in acc_files:
        for vtt_file in vtt_files:
            if acc_file.split(".")[0] == vtt_file.split(".")[0]:
                pairs.append((acc_file, vtt_file))

    print(f"{output_dir}")

    for p in pairs:
        # split filename with ., drop last element, recompile as string back
        episode_name = ".".join(p[0].split(".")[:-1])
        print(p, episode_name)

        episode_dir = output_dir / episode_name
        episode_dir.mkdir(exist_ok=True)
        # print(f'{input_dir}/{p[1]}',f'{input_dir}/{p[0]}')
        print(episode_dir)
        split_audio_with_subtitles(f"{input_dir}/{p[1]}", f"{input_dir}/{p[0]}", episode_dir, tolerance=tolerance, period_threshold=pt, force_merge=force_merge, forced_alignment=forced_alignment)


@cli.command()
@click.argument("input_", metavar="INPUT")
@click.option(
    "-r",
    "--reverse",
    is_flag=True,
    help="Reverse the input.",
)
def repeat(input_: str, *, reverse: bool = False) -> None:
    """Repeat the input."""
    click.echo(input_ if not reverse else input_[::-1])
