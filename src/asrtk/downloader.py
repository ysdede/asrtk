ydl_opts = {
    # ℹ️ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
    'format': 'm4a/bestaudio/best',
    'postprocessors': [{  # Extract audio using ffmpeg
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
