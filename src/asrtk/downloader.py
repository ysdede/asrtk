ydl_opts = {
    # ℹ️ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
    'format': 'm4a/bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'aac',
    }],
    'writesubtitles': True,
    'subtitleslangs': ['tr'],
    'writeautomaticsub': False,
    'paths': {'home': 'raw_audio'},
    'restrictfilenames': True,
    'cachedir': 'temp',
    'concurrent_fragment_downloads': 1,
    'windowsfilenames': True,
    'ignoreerrors': True,
    'download_archive': 'downloaded.txt',
}

def vid_info(URL: str = ''):
    if not URL:
        return

    import yt_dlp

    ydl_opts = {'ignoreerrors': True}

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(URL, download=False)

        return ydl.sanitize_info(info)

def check_sub_lang(URL: str = '', info: dict = {}, lang: str = 'tr'):
    if not (URL or info):
        print('No URL or info provided')
        return

    if not info:
        info = vid_info(URL)

    if 'subtitles' in info:
        return lang in info['subtitles']

def check_audio_lang(info: dict = {}, lang: str = 'tr') -> bool:
    """Check if video has audio in specified language."""
    if not info:
        return False

    # Check audio language in different possible locations
    audio_lang = None

    # Check in format-specific information
    formats = info.get('formats', [])
    for format in formats:
        if format.get('language'):
            audio_lang = format.get('language')
            break

    # Check in general video information
    if not audio_lang:
        audio_lang = info.get('language')

    # Check automatic captions language as fallback
    # (often indicates original audio language)
    if not audio_lang and 'automatic_captions' in info:
        auto_caps_langs = info['automatic_captions'].keys()
        if lang in auto_caps_langs:
            audio_lang = lang

    return audio_lang == lang if audio_lang else False

def check_video_langs(info: dict = {}, lang: str = 'tr') -> tuple[bool, bool]:
    """Check if video has both subtitles and audio in specified language."""
    has_subs = check_sub_lang(info=info, lang=lang)
    has_audio = check_audio_lang(info=info, lang=lang)
    return has_subs, has_audio
