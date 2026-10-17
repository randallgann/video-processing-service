import os

import pytest

import ytdlp_opts
from ytdlp_opts import build_ydl_opts, proxy_for_attempt

# The option dict yt-dlp-aduio-processor-v1.py built inline on main @ 53380e8, minus
# proxy/cookie keys. The audio path must keep sending exactly this.
MAIN_AUDIO_OPTS = {
    'format': 'bestaudio/best',
    'restrictfilenames': True,
    'quiet': False,
    'no_warnings': False,
    'extract_flat': False,
    'remote_components': ['ejs:github'],
    'extractor_args': {'youtube': {'player_client': ['android', 'ios', 'tv']}},
    'socket_timeout': 60,
    'retries': 10,
    'fragment_retries': 10,
    'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
}
SUBTITLE_KEYS = ('skip_download', 'writesubtitles', 'writeautomaticsub', 'subtitlesformat', 'subtitleslangs')


@pytest.fixture
def no_proxy(monkeypatch, tmp_path):
    monkeypatch.delenv('YTDLP_PROXY', raising=False)
    monkeypatch.setenv('YOUTUBE_COOKIES_FILE', str(tmp_path / 'absent.txt'))


def test_proxy_set_uses_proxy_and_no_cookie_jar(monkeypatch, tmp_path):
    cookies = tmp_path / 'cookies.txt'
    cookies.write_text('# Netscape HTTP Cookie File\n')
    monkeypatch.setenv('YOUTUBE_COOKIES_FILE', str(cookies))
    monkeypatch.setenv('YTDLP_PROXY', 'http://user-rotate:pw@p.example:80')
    opts = build_ydl_opts(str(tmp_path / 'work'), audio=True)
    assert opts['proxy'] == 'http://user-rotate:pw@p.example:80'
    assert 'cookiefile' not in opts and 'user_agent' not in opts


def test_no_proxy_copies_cookies_into_the_work_dir(monkeypatch, tmp_path):
    cookies = tmp_path / 'cookies.txt'
    cookies.write_text('# Netscape HTTP Cookie File\n')
    work = tmp_path / 'work'
    work.mkdir()
    monkeypatch.delenv('YTDLP_PROXY', raising=False)
    monkeypatch.setenv('YOUTUBE_COOKIES_FILE', str(cookies))
    opts = build_ydl_opts(str(work), audio=False)
    assert opts['cookiefile'] == str(work / 'yt-cookies.txt')
    assert (work / 'yt-cookies.txt').read_text() == cookies.read_text()
    assert opts['user_agent'].startswith('Mozilla/5.0')
    assert 'proxy' not in opts


def test_audio_matches_main(no_proxy, tmp_path):
    work = str(tmp_path)
    opts = build_ydl_opts(work, audio=True)
    assert opts == dict(MAIN_AUDIO_OPTS, paths={'home': work},
                        outtmpl={'default': os.path.join(work, '%(title)s.%(ext)s')})
    assert not any(key in opts for key in SUBTITLE_KEYS)


def test_captions_options(no_proxy, tmp_path):
    opts = build_ydl_opts(str(tmp_path), audio=False, lang='es')
    assert opts['skip_download'] is True
    assert opts['writesubtitles'] is True and opts['writeautomaticsub'] is True
    assert opts['subtitlesformat'] == 'json3'
    assert opts['subtitleslangs'] == ['es', 'es-.*']
    assert 'postprocessors' not in opts and 'format' not in opts
    # Same network behaviour as the audio path.
    for key in ('extractor_args', 'remote_components', 'socket_timeout', 'retries', 'fragment_retries'):
        assert opts[key] == MAIN_AUDIO_OPTS[key]


def test_proxy_for_attempt_pins_and_walks_the_pool(monkeypatch):
    monkeypatch.setattr(ytdlp_opts, 'PROXY_START', 0)
    monkeypatch.setenv('YTDLP_PROXY_POOL_SIZE', '3')
    proxy = 'http://user-rotate:pw@p.example:80'
    assert [proxy_for_attempt(proxy, a) for a in (1, 2, 3, 4)] == [
        'http://user-1:pw@p.example:80', 'http://user-2:pw@p.example:80',
        'http://user-3:pw@p.example:80', 'http://user-1:pw@p.example:80']
    assert proxy_for_attempt('http://fixed:pw@p.example:80', 1) == 'http://fixed:pw@p.example:80'


def test_downloader_uses_the_shared_module():
    with open(os.path.join(os.path.dirname(ytdlp_opts.__file__), 'yt-dlp-aduio-processor-v1.py')) as f:
        source = f.read()
    assert 'from ytdlp_opts import build_ydl_opts, extract_with_bot_check_retry' in source
    assert 'def extract_with_bot_check_retry' not in source and "'postprocessors'" not in source
