import os
import re

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


@pytest.fixture(autouse=True)
def chain_state(monkeypatch, tmp_path):
    """Each test gets its own proxy-chain state file; a cooldown must not carry into the next test."""
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    monkeypatch.delenv('YTDLP_PROXY_CHAIN', raising=False)


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
    assert opts['subtitleslangs'] == ['es']
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


# Live-check rework: only the picked track is ever requested.
@pytest.mark.parametrize('code', ['en', 'en-US', 'en-orig', 'zh-Hant', 'pt-BR'])
def test_captions_request_exactly_the_picked_code(no_proxy, tmp_path, code):
    opts = build_ydl_opts(str(tmp_path), audio=False, lang=code)
    assert opts['subtitleslangs'] == [re.escape(code)]
    [pattern] = opts['subtitleslangs']
    # yt-dlp matches entries as regexes against every listed track: only `code` may match.
    listed = [code, f'{code}-en', f'{code}-de', 'en-en', 'en-de', 'en-US', 'en', 'en-orig']
    assert [c for c in listed if re.compile(pattern + '$').match(c)] == [c for c in listed if c == code]


def test_probe_opts_write_nothing(no_proxy, tmp_path):
    opts = ytdlp_opts.build_probe_opts(str(tmp_path))
    for key in ('writesubtitles', 'writeautomaticsub', 'subtitlesformat', 'subtitleslangs', 'postprocessors', 'format'):
        assert key not in opts
    assert opts['extractor_args'] == MAIN_AUDIO_OPTS['extractor_args']


def test_subtitle_429_uses_the_full_proxy_retry_budget(monkeypatch):
    """yt-dlp surfaces a timedtext 429 as DownloadError('... Unable to download video subtitles ...: HTTP Error 429: Too Many Requests')."""
    monkeypatch.delenv('YTDLP_BOT_CHECK_ATTEMPTS', raising=False)
    monkeypatch.setattr(ytdlp_opts.time, 'sleep', lambda s: None)
    proxies = []

    class FakeYDL:
        failures = 9

        def __init__(self, opts):
            proxies.append(opts['proxy'])

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download):
            if FakeYDL.failures:
                FakeYDL.failures -= 1
                raise ytdlp_opts.yt_dlp.utils.DownloadError(
                    "ERROR: Unable to download video subtitles for 'en': HTTP Error 429: Too Many Requests")
            return {'id': 'x'}

        def prepare_filename(self, info):
            return 'x'

    monkeypatch.setattr(ytdlp_opts.yt_dlp, 'YoutubeDL', FakeYDL)
    monkeypatch.setattr(ytdlp_opts, 'PROXY_START', 0)
    info, _ = ytdlp_opts.extract_with_bot_check_retry({'proxy': 'http://user-rotate:pw@p.example:80'}, 'u', proxied=True)
    assert info == {'id': 'x'}
    assert len(proxies) == 10, 'all 10 attempts, each on a different pinned proxy'
    assert len(set(proxies)) == 10

    FakeYDL.failures = 99
    proxies.clear()
    with pytest.raises(ytdlp_opts.yt_dlp.utils.DownloadError):
        ytdlp_opts.extract_with_bot_check_retry({'proxy': 'http://user-rotate:pw@p.example:80'}, 'u', proxied=True)
    assert len(proxies) == 10


def test_too_many_requests_without_the_status_code_is_retryable():
    assert any(m in 'Unable to download video subtitles: Too Many Requests' for m in ytdlp_opts.RETRYABLE_MARKERS)


def _failing_ydl(monkeypatch, proxies):
    class FakeYDL:
        def __init__(self, opts):
            proxies.append(opts.get('proxy'))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download):
            raise ytdlp_opts.yt_dlp.utils.DownloadError(
                "ERROR: Unable to download video subtitles for 'en': HTTP Error 429: Too Many Requests")

        def prepare_filename(self, info):
            return 'x'

    monkeypatch.setattr(ytdlp_opts.yt_dlp, 'YoutubeDL', FakeYDL)
    monkeypatch.setattr(ytdlp_opts.time, 'sleep', lambda s: None)
    monkeypatch.setattr(ytdlp_opts, 'PROXY_START', 0)


def test_attempts_override_makes_exactly_one_attempt_on_a_proxied_429(monkeypatch):
    """Captions fail-fast: attempts=1 raises after ONE attempt even with the env budget at 10."""
    monkeypatch.setenv('YTDLP_BOT_CHECK_ATTEMPTS', '10')
    proxies = []
    _failing_ydl(monkeypatch, proxies)
    with pytest.raises(ytdlp_opts.yt_dlp.utils.DownloadError):
        ytdlp_opts.extract_with_bot_check_retry({'proxy': 'http://user-rotate:pw@p.example:80'}, 'u', proxied=True, attempts=1)
    assert len(proxies) == 1


def test_without_the_override_the_env_budget_still_applies(monkeypatch):
    monkeypatch.setenv('YTDLP_BOT_CHECK_ATTEMPTS', '3')
    proxies = []
    _failing_ydl(monkeypatch, proxies)
    with pytest.raises(ytdlp_opts.yt_dlp.utils.DownloadError):
        ytdlp_opts.extract_with_bot_check_retry({'proxy': 'http://user-rotate:pw@p.example:80'}, 'u', proxied=True)
    assert len(proxies) == 3


@pytest.mark.parametrize('attempts, expected', [(2, 2), (0, 1)])
def test_attempts_override_is_floored_at_one_and_unproxied_is_one(monkeypatch, attempts, expected):
    proxies = []
    _failing_ydl(monkeypatch, proxies)
    with pytest.raises(ytdlp_opts.yt_dlp.utils.DownloadError):
        ytdlp_opts.extract_with_bot_check_retry({'proxy': 'http://user-rotate:pw@p.example:80'}, 'u', proxied=True, attempts=attempts)
    assert len(proxies) == expected
    proxies.clear()
    with pytest.raises(ytdlp_opts.yt_dlp.utils.DownloadError):
        ytdlp_opts.extract_with_bot_check_retry({}, 'u', proxied=False, attempts=5)
    assert len(proxies) == 1


def test_back_compat_ytdlp_proxy_alone_makes_the_same_attempts_as_before_the_chain(monkeypatch):
    """Issue #3: no YTDLP_PROXY_CHAIN -> one webshare provider; the URLs are 79143f9's sequence."""
    proxy = 'http://user-rotate:pw@p.example:80'
    monkeypatch.setenv('YTDLP_PROXY', proxy)
    monkeypatch.setenv('YTDLP_PROXY_POOL_SIZE', '3')
    monkeypatch.setenv('YTDLP_BOT_CHECK_ATTEMPTS', '3')
    proxies = []
    _failing_ydl(monkeypatch, proxies)
    with pytest.raises(ytdlp_opts.yt_dlp.utils.DownloadError):
        ytdlp_opts.extract_with_bot_check_retry({'proxy': proxy}, 'u', proxied=True)
    assert proxies == [proxy_for_attempt(proxy, a) for a in (1, 2, 3)] == [
        'http://user-1:pw@p.example:80', 'http://user-2:pw@p.example:80', 'http://user-3:pw@p.example:80']
