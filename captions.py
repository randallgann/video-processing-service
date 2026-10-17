"""
Captions-first transcripts.

For `quality: standard` the worker tries YouTube's own caption track before paying for
Whisper. Captions are turned into exactly the transcript JSON the Whisper path writes
(window_segments is shared with combine_transcription_results), so rag-widget's import
and the transcript cache see one format whichever source produced it.

Everything except fetch_captions is pure and offline-testable.
"""
import glob
import json
import logging
import os

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 60.0  # 1 minute chunks
# Rolling-caption overlap shorter than this many words is treated as natural speech
# ("... to the" / "the store"), not a repeated line.
MIN_OVERLAP_WORDS = 2


def window_segments(all_segments, video_info):
    """
    Group timed segments into ~60 s windows and wrap them in the transcript entry shape.

    Extracted unchanged from combine_transcription_results so captions and Whisper
    produce byte-identical output for the same segments.
    """
    # Sort segments by start time
    all_segments = sorted(all_segments, key=lambda x: x["start"])

    # Create 1-minute chunks for better context
    segments_for_vector = []
    current_chunk = []
    current_start = None
    current_end = None
    chunk_duration = WINDOW_SECONDS

    for seg in all_segments:
        if not current_chunk:
            current_start = seg["start"]
        current_chunk.append(seg["text"].strip())
        current_end = seg["end"]

        if (current_end - current_start) >= chunk_duration:
            chunk_text = " ".join(current_chunk)
            segments_for_vector.append({
                "start_time": current_start,
                "end_time": current_end,
                "text": chunk_text
            })
            current_chunk = []
            current_start = None

    # Add leftover chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        segments_for_vector.append({
            "start_time": current_start,
            "end_time": current_end,
            "text": chunk_text
        })

    logger.info(f"Created {len(segments_for_vector)} vector-ready segments")

    # Prepare final output
    final_output = []

    for segment in segments_for_vector:
        final_output.append({
            "text": f"Spoken Words: {segment['text']}",
            "metadata": {
                "title": video_info.get("title", ""),
                "video_url": video_info.get("url", ""),
                "video_id": video_info.get("id", ""),
                "timestamp_start": segment["start_time"],
                "timestamp_end": segment["end_time"],
                "upload_date": video_info.get("upload_date", "")
            }
        })

    return final_output


def _json3_formats(track):
    return [f for f in (track or []) if f.get('ext') == 'json3' and f.get('url')]


def _is_machine_translation(track):
    # yt-dlp lists every auto-translated language under automatic_captions; those URLs
    # carry tlang=. A translation of another language's ASR is not this video's speech.
    return any('tlang=' in f.get('url', '') for f in track or [])


def pick_track(info, lang):
    """
    Choose the caption track for `lang`.

    Precedence: manual exact, manual regional variant (en-US for en), automatic exact,
    automatic variant (e.g. en-orig). Machine-translated automatic tracks and tracks
    without a json3 format are skipped.

    Returns (language_code, kind) with kind in ('manual', 'auto'), or None.
    """
    lang = (lang or 'en').lower()
    for kind, key in (('manual', 'subtitles'), ('auto', 'automatic_captions')):
        tracks = info.get(key) or {}
        exact = [code for code in tracks if code.lower() == lang]
        variants = sorted(code for code in tracks if code.lower().startswith(f'{lang}-'))
        for code in exact + variants:
            track = tracks[code]
            if not _json3_formats(track):
                continue
            if kind == 'auto' and _is_machine_translation(track):
                continue
            return code, kind
    return None


def normalize_language(code):
    """'en-orig' -> 'en'; keeps real regional codes like 'en-US'."""
    if code and code.endswith('-orig'):
        return code[:-len('-orig')]
    return code


def json3_to_cues(json3):
    """Flatten json3 events into [{start, end, text}] seconds, dropping empty cues."""
    cues = []
    for event in json3.get('events') or []:
        segs = event.get('segs')
        if not segs:
            continue
        text = ''.join(seg.get('utf8', '') for seg in segs)
        text = ' '.join(text.split())
        if not text:
            continue
        start = event.get('tStartMs', 0) / 1000.0
        end = start + event.get('dDurationMs', 0) / 1000.0
        cues.append({"start": start, "end": end, "text": text})
    return cues


def _overlap_words(prev_words, next_words):
    """Longest k such that prev's last k words equal next's first k words."""
    for k in range(min(len(prev_words), len(next_words)), 0, -1):
        if prev_words[-k:] == next_words[:k]:
            return k
    return 0


def dedupe_rolling(cues):
    """
    Remove the rolling-caption artefact where each cue repeats (part of) the previous one.

    Each cue is compared word by word with the previous cue AS DISPLAYED (its original
    text), not with what was kept of it, so a chain like "a b c d" -> "c d e f" ->
    "c d e f g" still yields every word once ("a b c d", "e f g"):
    - next equals prev or extends it ("hello" -> "hello world"): its new words are
      appended to the last kept cue, which now spans both;
    - next is a shorter repeat of prev's opening (2+ words): dropped, span extended;
    - prev's trailing words repeated at next's head (2+ words, "a b c d" -> "c d e f"):
      next keeps only its new words ("e f");
    - anything else is distinct speech and passes through unchanged.
    Pure and idempotent.
    """
    out = []
    shown = None  # words of the previous cue as displayed
    for cue in cues:
        cue = dict(cue)
        next_words = cue["text"].split()
        if out:
            last = out[-1]
            if len(next_words) >= len(shown) and next_words[:len(shown)] == shown:
                added = next_words[len(shown):]
                if added:
                    last["text"] = ' '.join(last["text"].split() + added)
                last["end"] = max(last["end"], cue["end"])
                shown = next_words
                continue
            # A shorter cue repeating prev's opening; one word is too weak a signal.
            if MIN_OVERLAP_WORDS <= len(next_words) < len(shown) and shown[:len(next_words)] == next_words:
                last["end"] = max(last["end"], cue["end"])
                continue
            k = _overlap_words(shown, next_words)
            if k >= MIN_OVERLAP_WORDS:
                remainder = next_words[k:]
                shown = next_words
                if not remainder:
                    last["end"] = max(last["end"], cue["end"])
                    continue
                cue["text"] = ' '.join(remainder)
            else:
                shown = next_words
        else:
            shown = next_words
        out.append(cue)
    return out


def format_upload_date(upload_date):
    """yt-dlp's YYYYMMDD -> the MM-DD-YYYY the downloader's description file uses."""
    if upload_date and len(upload_date) == 8 and upload_date.isdigit():
        return f"{upload_date[4:6]}-{upload_date[6:8]}-{upload_date[:4]}"
    return upload_date or ""


def cues_to_segments(cues, video_info):
    """Captions cues -> the transcript entries combine_transcription_results writes."""
    return window_segments(cues, video_info)


def _caption_file(info, code, work_dir):
    entry = (info.get('requested_subtitles') or {}).get(code) or {}
    path = entry.get('filepath')
    if path and os.path.exists(path) and path.endswith('.json3'):
        return path
    matches = glob.glob(os.path.join(glob.escape(work_dir), f'*.{code}.json3'))
    return matches[0] if matches else None


def fetch_captions(url, lang='en', *, work_dir, info=None):
    """
    Download the video's caption track and convert it to transcript entries.

    The track is chosen from the video's metadata first (`info`, the worker's probe, or
    a metadata-only extraction when absent), then ONLY that exact track is downloaded:
    one timedtext request per video, none when there is no usable track.

    Returns {segments, language, kind, duration_seconds, title, channel_title,
    upload_date}, or None when there is no usable track. Never raises: any failure is
    logged and returns None so the caller falls back to Whisper.
    """
    from ytdlp_opts import build_probe_opts, build_ydl_opts, extract_with_bot_check_retry

    try:
        if info is None:
            probe_opts = build_probe_opts(work_dir)
            info, _ = extract_with_bot_check_retry(probe_opts, url, proxied=bool(probe_opts.get('proxy')), download=False)
        picked = pick_track(info, lang)
        if not picked:
            logger.info(f"No usable '{lang}' caption track for {url}")
            return None
        code, kind = picked
        logger.info(f"Fetching {kind} caption track {code} for {url}")
        opts = build_ydl_opts(work_dir, audio=False, lang=code)
        info, _ = extract_with_bot_check_retry(opts, url, proxied=bool(opts.get('proxy')))
        path = _caption_file(info, code, work_dir)
        if not path:
            logger.warning(f"Caption track {code} ({kind}) was listed but no json3 file was written")
            return None
        with open(path, 'r', encoding='utf-8') as f:
            cues = dedupe_rolling(json3_to_cues(json.load(f)))
        video_info = {
            "url": url,
            "id": info.get('id', ''),
            "title": info.get('title') or '',
            "upload_date": format_upload_date(info.get('upload_date')),
        }
        segments = cues_to_segments(cues, video_info)
        if not segments:
            logger.info(f"Caption track {code} ({kind}) for {url} has no text")
            return None
        duration = info.get('duration')
        return {
            "segments": segments,
            "language": normalize_language(code),
            "kind": kind,
            "duration_seconds": int(round(duration)) if duration else None,
            "title": info.get('title'),
            "channel_title": info.get('channel') or info.get('uploader'),
            "upload_date": video_info["upload_date"],
        }
    except Exception as e:
        logger.warning(f"Caption fetch failed for {url}, falling back to Whisper: {e}")
        return None
