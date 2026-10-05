import yt_dlp
import os
import random
import time
import urllib.parse
import shutil

# Failures that depend on which proxy exit IP we got, so a different IP can succeed:
# YouTube's bot check, a media URL rejected for the requesting IP, and rate limiting.
RETRYABLE_MARKERS = (
    "confirm you\u2019re not a bot",
    "HTTP Error 403",
    "HTTP Error 429",
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


def extract_with_bot_check_retry(ydl_opts, url, proxied):
    """
    Download with yt-dlp, retrying failures that depend on the proxy exit IP.

    Returns (info, prepared_filename).

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
                info = ydl.extract_info(url, download=True)
                return info, ydl.prepare_filename(info)
        except yt_dlp.utils.DownloadError as e:
            message = str(e).replace("'", "\u2019")
            if not any(marker in message for marker in RETRYABLE_MARKERS) or attempt == attempts:
                raise
            print(f"Proxy-dependent failure on attempt {attempt}/{attempts} ({message[:80]}); retrying on another proxy")
            time.sleep(min(2 * attempt, 10))


def download_audio(url, download_path="./"):
    """
    Downloads YouTube audio using yt-dlp
    """

    # Ensure download_path is an absolute path
    download_path = os.path.abspath(download_path)
    print(f"Using download path: {download_path}")

    # Configure yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',  # Download best quality
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
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192'
        }]
    }

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

    try:
        # Download the video
        print("Extracting video information...")
        info, base_filename = extract_with_bot_check_retry(ydl_opts, url, proxied=bool(proxy))
        title = info['title']
        # Get the base filename and ensure it's in the correct output directory
        # Make sure we're using the full path
        if not os.path.isabs(base_filename):
            base_filename = os.path.join(download_path, os.path.basename(base_filename))
        base_filename_no_ext, _ = os.path.splitext(base_filename)
        audio_path = base_filename_no_ext + ".mp3"
        print(f"Expected audio path: {audio_path}")
        
        # Verify file exists
        if os.path.exists(audio_path):
            print(f"Audio file exists at: {audio_path}")
        else:
            print(f"Audio file not found at: {audio_path}")
            # Try to find the file in the download directory
            for file in os.listdir(download_path):
                if file.endswith('.mp3'):
                    audio_path = os.path.join(download_path, file)
                    print(f"Found audio file at: {audio_path}")
                    break
        description = info.get('description', '')
        upload_date = info.get('upload_date')
        print(f"Download completed: {audio_path}")
        return audio_path, title, info['title'], description, upload_date
    except Exception as e:
        print(f"Download error: {str(e)}")
        raise

# def extract_audio(video_path, title, cleaned_title, download_path="./"):
#     """Extract audio from video file"""
#     base_path = os.path.splitext(video_path)[0]
#     audio_path = os.path.join(download_path, f"{base_path}.mp3")
#     try:
#         video = VideoFileClip(video_path)
#         audio = video.audio
#         print(f"Extracting audio to: {audio_path}")
#         audio.write_audiofile(audio_path)
#         video.close()
#         audio.close()
#         return audio_path
#     except Exception as e:
#         print(f"Audio extraction error: {str(e)}")
#         raise


def download_and_process_audio(url, download_path="./"):
    """
    Downloads a YouTube video and transcribes it with speaker identification
    """
    try:
        # Ensure download_path exists
        download_path = os.path.abspath(download_path)
        os.makedirs(download_path, exist_ok=True)
        print(f"Downloading audio to: {download_path}")
        
        # Download video using yt-dlp
        # video_path, video_title, cleaned_title = download_video(url, download_path)
        # print(f"Video saved as: {video_path}")

        # Download best audio using yt-dlp
        audio_path, audio_title, cleaned_title, description, upload_date = download_audio(url, download_path)
        print(f"Audio saved as: {audio_path}")

        # Extract episode number from audio_title
        # We'll iterate from the start until we hit a non-digit character
        episode_number = ""
        for char in audio_title:
            if char.isdigit():
                episode_number += char
            else:
                break

        # Now find the first alphabetic character after the digits to define where the cleaned title should start
        # If none found, we just use whatever is left
        import re
        alpha_match = re.search('[A-Za-z]', audio_title)
        if alpha_match:
            start_idx = alpha_match.start()
            final_cleaned_title = audio_title[start_idx:].strip()
        else:
            # No alpha found, just use the remainder
            final_cleaned_title = audio_title.strip()

        # Convert upload_date if present
        formatted_date = None
        if upload_date:
            # upload_date format is typically YYYYMMDD
            year = upload_date[:4]
            month = upload_date[4:6]
            day = upload_date[6:8]
            formatted_date = f"{month}-{day}-{year}"

        # Save the description to a text file
        base_path = os.path.splitext(audio_path)[0]
        desc_file_path = f"{base_path}.txt"
        with open(desc_file_path, "w", encoding="utf-8") as desc_file:
            if formatted_date:
                desc_file.write(f"Upload Date: {formatted_date}\n")
            if episode_number:
                desc_file.write(f"Episode_number: {episode_number}\n")
            desc_file.write(f"Title: {final_cleaned_title}\n\n")
            desc_file.write(description)
        print(f"Description saved as: {desc_file_path}")
        
        # Extract audio
        # print("Extracting audio...")
        # audio_path = extract_audio(video_path, video_title, cleaned_title, download_path)
        # print(f"Audio saved as: {audio_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Download audio from YouTube videos")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--output", default="./", help="Output directory (default: current directory)")
    args = parser.parse_args()
    
    download_and_process_audio(args.url, args.output)