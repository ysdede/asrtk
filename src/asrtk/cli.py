"""Main CLI for asrtk."""
import rich_click as click
import json
import re
from pathlib import Path
import yt_dlp
from .utils import is_valid_json_file
from .downloader import vid_info, check_sub_lang, ydl_opts

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


@cli.command()
@click.argument("work_dir", type=click.Path(exists=True))
@click.option("-u", "--url", "playlist_url", type=str, help="YouTube playlist URL")
@click.option("-f", "--file", "playlist_file", type=click.Path(exists=True), help="File containing playlist URLs")
def download_playlist(work_dir: str, playlist_url: str | None, playlist_file: str | None) -> None:
    """Download videos from a YouTube playlist with translated subtitles.

    Provide either a playlist URL using --url or a file containing URLs using --file
    """
    if not playlist_file and not playlist_url:
        raise click.UsageError("Either --url or --file must be specified.")
    if playlist_url and playlist_file:
        raise click.UsageError("Cannot specify both --url and --file. Choose one.")

    work_dir = Path(work_dir)
    json_info_dir = work_dir / "json_info"
    json_info_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts['paths']['home'] = str(work_dir)

    # Determine the source of playlist URLs
    if playlist_url:
        playlist_urls = [playlist_url]
    else:
        playlist_urls = [line.strip() for line in Path(playlist_file).read_text().splitlines()]

    for url in playlist_urls:
        if not url:
            continue  # Skip empty lines in the playlist file

        playlist_id = url.split('list=')[-1].split('&')[0]
        json_file = json_info_dir / f"{playlist_id}.json"

        if json_file.exists() and is_valid_json_file(json_file):
            click.echo(f"Loading existing playlist info from {json_file}")
            pl_info = json.loads(json_file.read_text())
        else:
            pl_info = vid_info(url)
            click.echo(f"Processing {pl_info['title']}, {pl_info['id']}, episodes: {len(pl_info['entries'])}...")
            json_file.write_text(json.dumps(pl_info, indent=4))

        click.echo(f"{pl_info['title']}, {pl_info['id']}")

        playlist_dir = work_dir / re.sub(r'[^a-zA-Z0-9]', '_', pl_info["title"])
        playlist_dir.mkdir(exist_ok=True)
        ydl_opts['paths']['home'] = str(playlist_dir)
        ydl_opts['download_archive'] = str(playlist_dir / 'downloaded.txt')

        for entry in pl_info['entries']:
            if check_sub_lang(info=entry):
                click.echo(f"{entry['title']}, {'✔' if check_sub_lang(info=entry) else '❌'}")

                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download(entry['id'])
                except Exception as e:
                    click.echo(f"An error occurred: {e}", err=True)
