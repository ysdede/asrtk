"""Commands for downloading YouTube videos with subtitles."""
from pathlib import Path
import rich_click as click
import yt_dlp
import json
import re

from ...utils.file import is_valid_json_file

def vid_info(url: str) -> dict:
    """Get video/playlist information from YouTube URL.

    Args:
        url: YouTube URL

    Returns:
        Video/playlist information dictionary
    """
    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
        return ydl.extract_info(url, download=False)

def check_sub_lang(info: dict, lang: str = "tr") -> bool:
    """Check if video has subtitles in specified language.

    Args:
        info: Video information dictionary
        lang: Language code to check

    Returns:
        True if subtitles are available
    """
    if not info.get('subtitles'):
        return False
    return lang in info['subtitles']

def check_audio_lang(info: dict, lang: str = "tr") -> bool:
    """Check if video has audio in specified language.

    Args:
        info: Video information dictionary
        lang: Language code to check

    Returns:
        True if audio is available
    """
    if not info.get('requested_formats'):
        return False
    for fmt in info['requested_formats']:
        if fmt.get('language') == lang:
            return True
    return False

def check_video_langs(info: dict, lang: str = "tr") -> tuple[bool, bool]:
    """Check if video has both subtitles and audio in specified language.

    Args:
        info: Video information dictionary
        lang: Language code to check

    Returns:
        Tuple of (has_subtitles, has_audio)
    """
    return check_sub_lang(info, lang), check_audio_lang(info, lang)

# YouTube-DL options for downloading
ydl_opts = {
    'format': 'bestaudio[language=tr]/bestaudio',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'm4a',
    }],
    'writesubtitles': True,
    'writeautomaticsub': False,
    'subtitleslangs': ['tr'],
    'skip_download': False,
    'quiet': True,
    'no_warnings': True,
    'ignoreerrors': True,
    'extract_flat': False,
    'paths': {'home': ''},  # Set dynamically
    'outtmpl': {
        'default': '%(title)s [%(id)s].%(ext)s',
        'subtitle': '%(title)s [%(id)s].%(ext)s'
    }
}

@click.command()
@click.argument("work_dir", type=click.Path(exists=True))
@click.option("-u", "--url", "playlist_url", type=str, help="YouTube playlist URL")
@click.option("-f", "--file", "playlist_file", type=click.Path(exists=True), help="File containing playlist URLs")
def download_playlist(work_dir: str, playlist_url: str | None, playlist_file: str | None) -> None:
    """Download videos from a YouTube playlist with subtitles.

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

    total_videos = 0
    downloaded = 0

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
            if not entry:  # Skip empty/private videos
                continue

            total_videos += 1
            video_id = entry.get('id') or entry.get('url', '').split('watch?v=')[-1]
            if not video_id:
                click.echo(f"Skipping entry with no video ID: {entry}")
                continue

            try:
                with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                    video_info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)

                has_subs, has_audio = check_video_langs(info=video_info)
                if has_subs and has_audio:
                    click.echo(f"Processing: {entry.get('title', video_id)}, ✔ (TR audio + subs)")
                    try:
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
                        downloaded += 1
                    except Exception as e:
                        click.echo(f"Error downloading {entry.get('title', video_id)}: {e}", err=True)
                else:
                    status = []
                    if not has_subs: status.append("no TR subs")
                    if not has_audio: status.append("no TR audio")
                    click.echo(f"Skipping: {entry.get('title', video_id)}, ❌ ({', '.join(status)})")

            except Exception as e:
                click.echo(f"Error checking video {video_id}: {e}", err=True)
                continue

    click.echo(f"\nDownload complete. {downloaded}/{total_videos} videos processed.")

@click.command()
@click.argument("work_dir", type=click.Path(exists=True))
@click.option("-u", "--url", "channel_url", type=str, help="YouTube channel URL")
@click.option("-f", "--file", "channel_file", type=click.Path(exists=True), help="File containing channel info JSON")
def download_channel(work_dir: str, channel_url: str | None, channel_file: str | None) -> None:
    """Download videos from a YouTube channel with subtitles.

    Provide either a channel URL using --url or a pre-downloaded channel info file using --file
    """
    if not channel_file and not channel_url:
        raise click.UsageError("Either --url or --file must be specified.")
    if channel_url and channel_file:
        raise click.UsageError("Cannot specify both --url and --file. Choose one.")

    work_dir = Path(work_dir)
    json_info_dir = work_dir / "json_info"
    json_info_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts['paths']['home'] = str(work_dir)

    # Get channel info either from URL or file
    if channel_url:
        # Extract and sanitize channel ID/name
        if '/c/' in channel_url:
            channel_id = channel_url.split('/c/')[-1]
        elif '/@' in channel_url:
            channel_id = channel_url.split('/@')[-1]
        elif '/channel/' in channel_url:
            channel_id = channel_url.split('/channel/')[-1]
        else:
            channel_id = channel_url.split('/')[-1]

        # Remove URL parameters if any
        channel_id = channel_id.split('?')[0]
        # Sanitize the channel ID for filename
        channel_id = re.sub(r'[^a-zA-Z0-9_-]', '_', channel_id)
        json_file = json_info_dir / f"{channel_id}.json"

        if json_file.exists() and is_valid_json_file(json_file):
            click.echo(f"Loading existing channel info from {json_file}")
            channel_info = json.loads(json_file.read_text())
        else:
            # Configure yt-dlp options for channel extraction
            channel_opts = {
                'quiet': True,
                'extract_flat': True,
                'ignoreerrors': True,
                'dump_single_json': True,
                'playlistreverse': False,
                'playlistend': None,
                'simulate': True,
                'skip_download': True,
                'no_warnings': True,
                'extract_flat': 'in_playlist',
                'playlist_items': '1-1000',  # Adjust if needed for larger channels
            }

            # First get channel info and videos list
            with yt_dlp.YoutubeDL(channel_opts) as ydl:
                # Add /videos to channel URL to get all uploads
                channel_vids_url = f"{channel_url}/videos"
                channel_info = ydl.extract_info(channel_vids_url, download=False)

            click.echo(f"Saving channel info to {json_file}")
            json_file.write_text(json.dumps(channel_info, indent=4))
    else:
        channel_info = json.loads(Path(channel_file).read_text())

    # Create channel directory
    channel_dir = work_dir / re.sub(r'[^a-zA-Z0-9]', '_', channel_info["title"])
    channel_dir.mkdir(exist_ok=True)
    ydl_opts['paths']['home'] = str(channel_dir)
    ydl_opts['download_archive'] = str(channel_dir / 'downloaded.txt')

    # Process all entries/videos in the channel
    total_videos = 0
    downloaded = 0

    # Get all video entries
    entries = channel_info.get('entries', [])

    click.echo(f"Found {len(entries)} videos in channel {channel_info['title']}")

    for entry in entries:
        if not entry:  # Skip empty/private videos
            continue

        total_videos += 1
        video_id = entry.get('id') or entry.get('url', '').split('watch?v=')[-1]
        if not video_id:
            click.echo(f"Skipping entry with no video ID: {entry}")
            continue

        # Get full video info to check subtitles
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                video_info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)

            has_subs, has_audio = check_video_langs(info=video_info)

            if has_subs and has_audio:
                click.echo(f"Processing: {entry.get('title', video_id)}, ✔ (TR audio + subs)")
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
                    downloaded += 1
                except Exception as e:
                    click.echo(f"Error downloading {entry.get('title', video_id)}: {e}", err=True)
            else:
                status = []
                if not has_subs: status.append("no TR subs")
                if not has_audio: status.append("no TR audio")
                click.echo(f"Skipping: {entry.get('title', video_id)}, ❌ ({', '.join(status)})")

        except Exception as e:
            click.echo(f"Error checking video {video_id}: {e}", err=True)
            continue

    click.echo(f"\nDownload complete. {downloaded}/{total_videos} videos processed.")
