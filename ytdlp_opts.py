"""
yt-dlp options and the proxy-aware extract/retry loop, shared by the audio downloader
(yt-dlp-aduio-processor-v1.py, run as a subprocess) and the in-process captions path
(captions.py). Both must reach YouTube the same way: same player clients, same proxy
pinning, same cookie fallback, same 403/429/bot-check retry.
"""
import os
import random
import re
import shutil
import time
import urllib.parse

import yt_dlp

# Failures that depend on which proxy exit IP we got, so a different IP can succeed:
# YouTube's bot check, a media URL rejected for the requesting IP, and rate limiting.
RETRYABLE_MARKERS = (
    "confirm you’re not a bot",
    "HTTP Error 403",
    "HTTP Error 429",
    "Too Many Requests",
)


def proxy_for_attempt(proxy, attempt):
    """
    Pin one exit IP for the whole attempt.

    Webshare's `<user>-rotate` username switches exit IP on every request, but YouTube's
    media URLs are bound to the IP that extracted them, so extraction and download land
    on different IPs and the download fails with 403. `<user>-<n>` pins proxy n of the
    plan's pool (YTDLP_PROXY_POOL_SIZE, 10 on the current plan), so each attempt uses one
    IP throughout and successive attempts walk the pool.
    """
    parsed = urllib.parse.urlparse(proxy)
    if not parsed.username or not parsed.username.endswith('-rotate'):
        return proxy
    pool_size = int(os.environ.get('YTDLP_PROXY_POOL_SIZE', '10'))
    index = (PROXY_START + attempt - 1) % pool_size + 1
    user = parsed.username[:-len('rotate')] + str(index)
    netloc = f"{user}:{parsed.password}@{parsed.hostname}:{parsed.port}"
    return urllib.parse.urlunparse(parsed._replace(netloc=netloc))


PROXY_START = random.randrange(1000)


def extract_with_bot_check_retry(ydl_opts, url, proxied, download=True):
    """
    Download with yt-dlp, retrying failures that depend on the proxy exit IP.

    Returns (info, prepared_filename). With download=False only the metadata is
    extracted (the duration probe).

    Each attempt pins a different proxy from the pool (see proxy_for_attempt) and builds
    a NEW YoutubeDL so no pooled connection carries over. Measured in the pod
    2026-10-04: per-request rotation failed downloads with 403 (IP-bound media URLs) and
    some exit IPs hit the bot check, while others pass. Without a proxy the IP never
    changes, so retrying is pointless and we fail fast.
    """
    attempts = int(os.environ.get('YTDLP_BOT_CHECK_ATTEMPTS', '10')) if proxied else 1
    for attempt in range(1, attempts + 1):
        opts = dict(ydl_opts)
        if proxied:
            opts['proxy'] = proxy_for_attempt(ydl_opts['proxy'], attempt)
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=download)
                return info, ydl.prepare_filename(info)
        except yt_dlp.utils.DownloadError as e:
            message = str(e).replace("'", "’")
            if not any(marker in message for marker in RETRYABLE_MARKERS) or attempt == attempts:
                raise
            print(f"Proxy-dependent failure on attempt {attempt}/{attempts} ({message[:80]}); retrying on another proxy")
            time.sleep(min(2 * attempt, 10))


def build_ydl_opts(download_path, *, audio, lang='en'):
    """
    yt-dlp options for one job.

    audio=True: download the best audio and convert it to mp3 (the Whisper path).
    audio=False: download nothing but the json3 caption track whose code is exactly
    `lang` — the code captions.pick_track already chose (e.g. 'en-US'). yt-dlp treats
    subtitleslangs entries as regexes, so the code is escaped: one track, one timedtext
    request, never the auto-translated variants a pattern like 'en-.*' would also match.
    """
    ydl_opts = {
        'paths': {'home': download_path},  # Set the base download path
        'outtmpl': {'default': os.path.join(download_path, '%(title)s.%(ext)s')},
        'quiet': False,  # Show progress
        'no_warnings': False,
        'extract_flat': False,
        'restrictfilenames': True,
        # Allow yt-dlp to fetch the EJS challenge-solver script (run via deno) so the
        # YouTube nsig "n challenge" can be solved; without it format extraction fails
        # with "Requested format is not available".
        'remote_components': ['ejs:github'],
        # The default `web` player client now fails YouTube's player challenge
        # ("The page needs to be reloaded"), independent of IP or cookies. The
        # `android` client extracts formats and captions reliably; list order is
        # the fallback order (yt-dlp tries each until one succeeds).
        'extractor_args': {'youtube': {'player_client': ['android', 'ios', 'tv']}},
        # Residential proxies are slower/flakier than a direct datacenter link, so
        # the default 20s read timeout trips mid-download of the googlevideo CDN
        # stream. Give it room and retry transient proxy drops.
        'socket_timeout': 60,
        'retries': 10,
        'fragment_retries': 10,
    }
    if audio:
        ydl_opts['format'] = 'bestaudio/best'  # Download best quality
        ydl_opts['postprocessors'] = [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192'
        }]
    else:
        ydl_opts.update({
            'skip_download': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitlesformat': 'json3',
            'subtitleslangs': [re.escape(lang)],
        })

    # Preferred anti-bot path: route through a residential proxy. YouTube blocks
    # datacenter IPs ("Sign in to confirm you're not a bot") regardless of player
    # client; a residential IP clears the challenge without any cookies. When a proxy
    # is configured we deliberately skip the cookie jar — stale/invalidated session
    # cookies can themselves re-trigger bot-detection.
    proxy = os.environ.get('YTDLP_PROXY')
    if proxy:
        ydl_opts['proxy'] = proxy
        # Redact credentials in logs.
        print(f"Routing yt-dlp through residential proxy {proxy.split('@')[-1]}")

    # Legacy cookie fallback (used only when no proxy is set — e.g. local/residential
    # runs). Mounted as a read-only secret, so copy it to a writable path first
    # (yt-dlp rewrites the cookie jar on exit).
    cookies_src = os.environ.get('YOUTUBE_COOKIES_FILE', '/var/secrets/youtube/cookies.txt')
    if not proxy and os.path.exists(cookies_src):
        cookies_rw = os.path.join(download_path, 'yt-cookies.txt')
        try:
            shutil.copyfile(cookies_src, cookies_rw)
            ydl_opts['cookiefile'] = cookies_rw
        except Exception as copy_err:
            print(f"WARNING: could not copy cookies to writable path ({copy_err}); using source")
            ydl_opts['cookiefile'] = cookies_src
        ydl_opts['user_agent'] = (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
            '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        print(f"Using YouTube cookies from {cookies_src}")
    elif not proxy:
        print(f"WARNING: no YTDLP_PROXY set and no cookies at {cookies_src}; "
              f"downloads will likely fail with bot-detection from datacenter IPs")

    return ydl_opts


def build_probe_opts(download_path):
    """Metadata-only extraction (duration, title, caption track listing); writes nothing."""
    opts = build_ydl_opts(download_path, audio=False)
    for key in ('writesubtitles', 'writeautomaticsub', 'subtitlesformat', 'subtitleslangs'):
        opts.pop(key, None)
    return opts
