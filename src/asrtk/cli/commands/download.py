"""Commands for downloading YouTube videos with subtitles."""
from pathlib import Path
import rich_click as click
import yt_dlp
import json
import re
import csv
from datetime import datetime
import os

def is_valid_json_file(file_path, size_threshold=1024):
    """Check if the file exists, has a size greater than the threshold and is a valid JSON."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) < size_threshold:
        return False
    try:
        with open(file_path, "r") as file:
            json.load(file)
        return True
    except json.JSONDecodeError:
        return False

def vid_info(url: str) -> dict:
    """Get video/playlist information from YouTube URL.

    Args:
        url: YouTube URL

    Returns:
        Video/playlist information dictionary
    """
    with yt_dlp.YoutubeDL({'quiet': False, 'ignoreerrors': True}) as ydl:
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
    # Check in all available formats
    formats = info.get('formats', [])
    for fmt in formats:
        if fmt.get('language') == lang:
            return True
            
    # Also check in requested_formats as backup
    requested_formats = info.get('requested_formats', [])
    for fmt in requested_formats:
        if fmt.get('language') == lang:
            return True

    # print(f"Audio in {lang} not found", formats, requested_formats)
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
    'subtitleslangs': ['en'],
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
@click.option("--all", is_flag=True, help="Do not check subtitles and audio languages")
def download_playlist(work_dir: str, playlist_url: str | None, playlist_file: str | None, all: bool) -> None:
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

                if all:
                    has_subs, has_audio = True, True
                else:
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
@click.option("--all", is_flag=True, help="Do not check subtitles and audio languages")
def download_channel(work_dir: str, channel_url: str | None, channel_file: str | None, all: bool) -> None:
    """Download videos from a YouTube channel with subtitles.

    Provide either a channel URL using --url or a pre-downloaded channel info file using --file.
    Use --all to download all videos regardless of subtitle and audio languages.
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
                'playlist_items': '1-100000',  # Adjust if needed for larger channels
            }

            # First get channel info and videos list
            with yt_dlp.YoutubeDL(channel_opts) as ydl:
                # Add /videos to channel URL to get all uploads
                channel_vids_url = f"{channel_url}/videos"
                channel_info = ydl.extract_info(channel_vids_url, download=False)

            click.echo(f"Saving channel info to {json_file}")
            json_file.write_text(json.dumps(channel_info, indent=4))
    else:
        if not is_valid_json_file(channel_file):
            raise click.BadParameter(f"Invalid JSON file: {channel_file}")
        click.echo("Reading channel info from file...")
        with open(channel_file, 'r', encoding='utf-8') as f:
            channel_info = json.load(f)

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

            if all:
                has_subs, has_audio = True, True
            else:
                has_subs, has_audio = check_video_langs(video_info)

            if has_subs and has_audio:
                click.echo(f"Processing: {entry.get('title', video_id)}, ✔ (TR audio + subs)")
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
                    downloaded += 1
                except Exception as e:
                    click.echo(f"Error downloading {entry.get('title', video_id)}: {e}")
            else:
                status = []
                if not has_subs: status.append("no TR subs")
                if not has_audio: status.append("no TR audio")
                click.echo(f"Skipping: {entry.get('title', video_id)}, ❌ ({', '.join(status)})")

        except Exception as e:
            click.echo(f"Error checking video {video_id}: {e}")
            continue

    click.echo(f"\nDownload complete. {downloaded}/{total_videos} videos processed.")

def load_processed_videos(work_dir: Path) -> dict:
    """Load all previously processed videos from existing log files.
    
    Returns a dict with video_id as key and dict containing status and metadata as value.
    """
    processed_videos = {}
    log_files = list(work_dir.glob("video_log_*.csv"))
    if log_files:
        click.echo(f"Found {len(log_files)} previous log files:")
    for log_file in log_files:
        try:
            with open(log_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    video_id = row.get('youtube_id')
                    if video_id:
                        processed_videos[video_id] = row
                        count += 1
                click.echo(f"  - {log_file.name}: {count} videos")
        except Exception as e:
            click.echo(f"Warning: Could not read log file {log_file}: {e}")
    return processed_videos

@click.command()
@click.argument("work_dir", type=click.Path(exists=True))
@click.option("-u", "--url", "channel_url", type=str, help="YouTube channel URL")
@click.option("-f", "--file", "channel_file", type=click.Path(exists=True), help="Pre-downloaded channel info file")
def download_channel_wosub(work_dir: str, channel_url: str | None, channel_file: str | None) -> None:
    """Download videos from a YouTube channel that have Turkish audio but NO Turkish subtitles.

    Provide either a channel URL using --url or a pre-downloaded channel info file using --file
    """
    if not channel_file and not channel_url:
        raise click.UsageError("Either --url or --file must be specified.")
    if channel_url and channel_file:
        raise click.UsageError("Cannot specify both --url and --file. Choose one.")

    work_dir = Path(work_dir)
    work_dir.mkdir(exist_ok=True)
    json_file = work_dir / 'channel_info.json'

    # Load previously processed videos
    click.echo("\nChecking for previously processed videos...")
    processed_videos = load_processed_videos(work_dir)
    if processed_videos:
        already_downloaded = sum(1 for v in processed_videos.values() if v.get('status') == 'downloaded')
        click.echo(f"Found {len(processed_videos)} previously processed videos ({already_downloaded} downloaded)\n")
    else:
        click.echo("No previously processed videos found.\n")

    # Get channel info either from file or YouTube
    if channel_url:
        if json_file.exists() and is_valid_json_file(json_file):
            click.echo(f"Loading existing channel info from {json_file}")
            info = json.loads(json_file.read_text())
        else:
            click.echo("Fetching channel information from YouTube...")
            # Configure yt-dlp options for channel extraction
            channel_opts = {
                'quiet': False,
                'extract_flat': True,
                'ignoreerrors': True,
                'dump_single_json': True,
                'playlistreverse': False,
                'playlistend': None,
                'simulate': True,
                'skip_download': True,
                'no_warnings': False,
                'extract_flat': 'in_playlist',
                'playlist_items': '1-100000',  # Adjust if needed for larger channels
            }

            # First get channel info and videos list
            with yt_dlp.YoutubeDL(channel_opts) as ydl:
                # Add /videos to channel URL to get all uploads
                if not channel_url.endswith('/videos'):
                    channel_vids_url = f"{channel_url}/videos"
                else:
                    channel_vids_url = channel_url
                
                info = ydl.extract_info(channel_vids_url, download=False)

            click.echo(f"Saving channel info to {json_file}")
            json_file.write_text(json.dumps(info, ensure_ascii=False, indent=4))
    else:
        if not is_valid_json_file(channel_file):
            raise click.BadParameter(f"Invalid JSON file: {channel_file}")
        click.echo("Reading channel info from file...")
        with open(channel_file, 'r', encoding='utf-8') as f:
            info = json.load(f)

    if 'entries' not in info:
        raise click.UsageError("No videos found in channel")

    total_videos = len([e for e in info['entries'] if e])
    downloaded = 0
    processed = 0
    click.echo(f"Found {total_videos} videos in channel")

    # Create a copy of ydl_opts and modify for video download
    video_opts = ydl_opts.copy()
    video_opts.update({
        'format': 'bestvideo[height<=480]+bestaudio[ext=m4a]/bestvideo[height<=480]+bestaudio/best[height<=480]',
        'merge_output_format': 'mp4',
        'writesubtitles': False,
        'writeautomaticsub': False,
        'postprocessors': [],  # Remove audio extraction postprocessor
        'quiet': False,  # Show progress
        'no_warnings': False,  # Show warnings
        'progress': True,  # Show progress bar
    })
    video_opts['paths']['home'] = str(work_dir)

    csv_file = work_dir / f"video_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_fields = ['title', 'youtube_id', 'url', 'duration', 'view_count', 'has_tr_audio', 'has_tr_subs', 'status', 'error']

    with yt_dlp.YoutubeDL(video_opts) as ydl:
        for entry in info['entries']:
            if not entry:
                continue

            video_id = entry.get('id', '')
            if not video_id:
                continue

            # Check if video was already processed
            if video_id in processed_videos:
                prev_data = processed_videos[video_id]
                prev_status = prev_data['status']
                
                # Only skip if it was successfully downloaded before
                if prev_status == 'downloaded':
                    click.echo(f"Already downloaded: {entry.get('title', video_id)}")
                    # Copy previous data to new log
                    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=csv_fields)
                        if f.tell() == 0:
                            writer.writeheader()
                        writer.writerow(processed_videos[video_id])
                    processed += 1
                    continue
                else:
                    click.echo(f"Rechecking: {entry.get('title', video_id)} (previous status: {prev_status})")

            try:
                # Get detailed video info
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                video_info = vid_info(video_url)
                has_subs, has_audio = check_video_langs(video_info)

                # Prepare video metadata
                video_data = {
                    'title': video_info.get('title', ''),
                    'youtube_id': video_id,
                    'url': video_url,
                    'duration': video_info.get('duration', 0),
                    'view_count': video_info.get('view_count', 0),
                    'has_tr_audio': has_audio,
                    'has_tr_subs': has_subs,
                    'status': '',
                    'error': ''
                }

                # We want videos with Turkish audio but NO Turkish subtitles
                if has_audio and not has_subs:
                    click.echo(f"Downloading: {entry.get('title', video_id)} ✓ (TR audio, no TR subs)")
                    try:
                        ydl.download([video_url])
                        downloaded += 1
                        video_data['status'] = 'downloaded'
                    except Exception as e:
                        error_msg = str(e)
                        click.echo(f"Error downloading {entry.get('title', video_id)}: {error_msg}")
                        video_data['status'] = 'error'
                        video_data['error'] = error_msg
                else:
                    status = []
                    if has_subs: status.append("has TR subs")
                    if not has_audio: status.append("no TR audio")
                    status_msg = ', '.join(status)
                    click.echo(f"Skipping: {entry.get('title', video_id)}, ❌ ({status_msg})")
                    video_data['status'] = f'skipped ({status_msg})'

                # Log video metadata to CSV
                with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=csv_fields)
                    if f.tell() == 0:
                        writer.writeheader()
                    writer.writerow(video_data)
                processed += 1

            except Exception as e:
                error_msg = str(e)
                click.echo(f"Error checking video {video_id}: {error_msg}")
                # Log error in CSV
                video_data = {
                    'title': entry.get('title', ''),
                    'youtube_id': video_id,
                    'url': video_url,
                    'duration': 0,
                    'view_count': 0,
                    'has_tr_audio': False,
                    'has_tr_subs': False,
                    'status': 'error',
                    'error': error_msg
                }
                with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=csv_fields)
                    if f.tell() == 0:
                        writer.writeheader()
                    writer.writerow(video_data)
                processed += 1
                continue

    click.echo(f"\nDownload complete. {downloaded} videos downloaded, {processed}/{total_videos} videos processed.")
    click.echo(f"Video metadata logged to: {csv_file}")
