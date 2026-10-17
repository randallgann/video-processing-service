import copy
import json
import os

import pytest

import captions
from captions import (cues_to_segments, dedupe_rolling, format_upload_date, json3_to_cues, normalize_language,
                      pick_track, window_segments)
from conftest import FIXTURES

VIDEO_INFO = {"url": "https://www.youtube.com/watch?v=abc123def45", "id": "abc123def45",
              "title": "Captioned", "upload_date": "10-17-2026"}


@pytest.fixture(scope='module')
def json3():
    with open(os.path.join(FIXTURES, 'sample.json3'), encoding='utf-8') as f:
        return json.load(f)


def cue(start, end, text):
    return {"start": start, "end": end, "text": text}


# 1. json3_to_cues
def test_json3_to_cues_on_fixture(json3):
    cues = json3_to_cues(json3)
    with_text = [e for e in json3['events'] if ''.join(s.get('utf8', '') for s in e.get('segs') or []).strip()]
    assert len(cues) == len(with_text) == 77
    assert cues[0] == cue(1.2, 4.66, 'the welcome automatic looking are are channel automatic')
    assert cues[-1]['end'] == pytest.approx(196.273)
    assert all(c['text'] and c['text'] == ' '.join(c['text'].split()) for c in cues)
    assert all(c['end'] >= c['start'] for c in cues)


def test_json3_to_cues_drops_header_newline_and_blank_events():
    doc = {"events": [
        {"tStartMs": 0, "dDurationMs": 9000},
        {"tStartMs": 100, "dDurationMs": 1000, "segs": [{"utf8": "hi"}, {"utf8": " there", "tOffsetMs": 300}]},
        {"tStartMs": 1090, "dDurationMs": 10, "aAppend": 1, "segs": [{"utf8": "\n"}]},
        {"tStartMs": 1200, "dDurationMs": 500, "segs": [{"utf8": "   "}]},
        {"tStartMs": 2000, "dDurationMs": 500, "segs": [{"utf8": "line one\nline two"}]},
    ]}
    assert json3_to_cues(doc) == [cue(0.1, 1.1, 'hi there'), cue(2.0, 2.5, 'line one line two')]
    assert json3_to_cues({}) == []


# 2. dedupe_rolling
@pytest.mark.parametrize('cues, expected', [
    pytest.param([cue(0, 2, 'hello world'), cue(2, 4, 'hello world')],
                 [cue(0, 4, 'hello world')], id='exact repeat'),
    pytest.param([cue(0, 1, 'hello'), cue(1, 3, 'hello world'), cue(3, 5, 'hello world again')],
                 [cue(0, 5, 'hello world again')], id='prefix growth'),
    pytest.param([cue(0, 3, 'the quick brown fox'), cue(3, 6, 'brown fox jumps over')],
                 [cue(0, 3, 'the quick brown fox'), cue(3, 6, 'jumps over')], id='tail/head overlap'),
    pytest.param([cue(0, 3, 'the quick brown fox'), cue(3, 5, 'brown fox')],
                 [cue(0, 5, 'the quick brown fox')], id='overlap with nothing new'),
    pytest.param([cue(0, 3, 'the quick brown fox'), cue(3, 4, 'the quick')],
                 [cue(0, 4, 'the quick brown fox')], id='shorter repeat of the opening'),
    pytest.param([cue(0, 2, 'I went to the'), cue(2, 4, 'the store today')],
                 [cue(0, 2, 'I went to the'), cue(2, 4, 'the store today')], id='one shared word is speech'),
    pytest.param([cue(0, 2, 'I think so'), cue(2, 3, 'I')],
                 [cue(0, 2, 'I think so'), cue(2, 3, 'I')], id='one-word repeat of the opening is speech'),
    pytest.param([cue(0, 1, 'helloworld'), cue(1, 2, 'hello')],
                 [cue(0, 1, 'helloworld'), cue(1, 2, 'hello')], id='word boundaries, not characters'),
    pytest.param([cue(0, 1, 'one two'), cue(1, 2, 'three four'), cue(2, 3, 'five six')],
                 [cue(0, 1, 'one two'), cue(1, 2, 'three four'), cue(2, 3, 'five six')], id='distinct no-op'),
    pytest.param([], [], id='empty'),
])
def test_dedupe_rolling_table(cues, expected):
    before = copy.deepcopy(cues)
    out = dedupe_rolling(cues)
    assert out == expected
    assert cues == before, 'input must not be mutated'
    assert dedupe_rolling(out) == out, 'idempotent'


def test_dedupe_rolling_on_fixture(json3):
    cues = json3_to_cues(json3)
    out = dedupe_rolling(cues)
    assert len(out) < len(cues)
    assert dedupe_rolling(out) == out
    for prev, nxt in zip(out, out[1:]):
        a, b = prev['text'].split(), nxt['text'].split()
        assert b[:len(a)] != a, 'no cue repeats or extends its predecessor'
    texts = [c['text'] for c in out]
    assert sum(t.endswith(' again') for t in texts) == 7, 'grown lines replace the line they repeat'
    assert 'and then more' in texts, 'overlapping head trimmed to the new words'


# 3. cues_to_segments
def test_cues_to_segments_shape_and_text(json3):
    cues = dedupe_rolling(json3_to_cues(json3))
    segments = cues_to_segments(cues, VIDEO_INFO)
    assert len(segments) == 4
    for seg in segments[:-1]:
        assert seg['metadata']['timestamp_end'] - seg['metadata']['timestamp_start'] >= 60
    for seg in segments:
        assert seg['text'].startswith('Spoken Words: ')
        assert set(seg) == {'text', 'metadata'}
        assert set(seg['metadata']) == {'title', 'video_url', 'video_id', 'timestamp_start', 'timestamp_end',
                                        'upload_date'}
        assert seg['metadata']['video_id'] == 'abc123def45'
    joined = ' '.join(s['text'][len('Spoken Words: '):] for s in segments)
    assert joined == ' '.join(c['text'] for c in cues)


# 4. window_segments regression against the pre-refactor combine (captured from main @ 53380e8)
def test_window_segments_matches_pre_refactor_combine(processor):
    with open(os.path.join(FIXTURES, 'whisper_results.json'), encoding='utf-8') as f:
        results = json.load(f)
    with open(os.path.join(FIXTURES, 'whisper_combined_expected.json'), encoding='utf-8') as f:
        expected = json.load(f)
    assert processor.combine_transcription_results(copy.deepcopy(results), expected['video_info']) == expected['output']
    assert json.dumps(processor.combine_transcription_results(copy.deepcopy(results), expected['video_info']),
                      indent=2) == json.dumps(expected['output'], indent=2)
    segs = [s for r in results if not r.get('failed') for s in r['segments']]
    assert window_segments(segs, expected['video_info']) == expected['output']


# 5. pick_track
def track(lang='en', translated=False, formats=('json3', 'vtt')):
    suffix = '&tlang=' + lang if translated else ''
    return [{'ext': ext, 'url': f'https://www.youtube.com/api/timedtext?v=x&lang={lang}&fmt={ext}{suffix}'}
            for ext in formats]


@pytest.mark.parametrize('info, lang, expected', [
    pytest.param({'subtitles': {'en': track()}, 'automatic_captions': {'en': track()}}, 'en', ('en', 'manual'),
                 id='manual beats auto'),
    pytest.param({'subtitles': {'en-US': track('en-US')}, 'automatic_captions': {'en': track()}}, 'en',
                 ('en-US', 'manual'), id='manual variant beats auto exact'),
    pytest.param({'subtitles': {'en-US': track('en-US'), 'en-GB': track('en-GB')}}, 'en', ('en-GB', 'manual'),
                 id='variant when exact is absent (stable order)'),
    pytest.param({'automatic_captions': {'en-orig': track(), 'en': track()}}, 'en', ('en', 'auto'),
                 id='auto exact'),
    pytest.param({'automatic_captions': {'en-orig': track()}}, 'en', ('en-orig', 'auto'), id='auto variant'),
    pytest.param({'automatic_captions': {'en': track(translated=True), 'es-orig': track('es'), 'es': track('es')}},
                 'en', None, id='machine translation is not this video\'s speech'),
    pytest.param({'subtitles': {'en': track(formats=('vtt', 'srv3'))}}, 'en', None, id='no json3 format'),
    pytest.param({'subtitles': {'fr': track('fr')}, 'automatic_captions': {}}, 'en', None, id='no matching track'),
    pytest.param({}, 'en', None, id='no tracks at all'),
    pytest.param({'subtitles': {'eng': track('eng')}}, 'en', None, id='prefix is by language tag, not characters'),
    pytest.param({'subtitles': {'es': track('es')}}, 'es', ('es', 'manual'), id='other language'),
])
def test_pick_track(info, lang, expected):
    assert pick_track(info, lang) == expected


def test_normalize_language_and_upload_date():
    assert normalize_language('en-orig') == 'en'
    assert normalize_language('en-US') == 'en-US'
    assert format_upload_date('20261017') == '10-17-2026'
    assert format_upload_date(None) == ''
    assert format_upload_date('2026') == '2026'


# fetch_captions glue (yt-dlp stubbed; no network)
def test_fetch_captions_downloads_only_the_picked_track(monkeypatch, tmp_path, json3):
    """Live-check rework: pick from metadata first, then one download of exactly that code."""
    written = tmp_path / 'Captioned.en-US.json3'
    written.write_text(json.dumps(json3), encoding='utf-8')
    listing = {'id': 'abc123def45', 'title': 'Captioned', 'duration': 196.4, 'subtitles': {'en-US': track('en-US')},
               'automatic_captions': {'en': track(), 'en-en': track(translated=True), 'en-de': track('de', translated=True)}}
    downloaded = dict(listing, requested_subtitles={'en-US': {'ext': 'json3', 'filepath': str(written)}})
    calls = []

    def fake_extract(opts, url, proxied, download=True, attempts=None):
        calls.append({'download': download, 'langs': opts.get('subtitleslangs'), 'write': opts.get('writesubtitles')})
        return (downloaded if download else listing), 'x'

    monkeypatch.setattr('ytdlp_opts.extract_with_bot_check_retry', fake_extract)

    # With the worker's probe info: exactly one request, for exactly the picked code.
    result = captions.fetch_captions(VIDEO_INFO['url'], 'en', work_dir=str(tmp_path), info=listing)
    assert result['language'] == 'en-US' and result['kind'] == 'manual'
    assert calls == [{'download': True, 'langs': ['en\\-US'], 'write': True}]

    # Without it: one metadata-only extraction (writes nothing), then the same single download.
    calls.clear()
    assert captions.fetch_captions(VIDEO_INFO['url'], 'en', work_dir=str(tmp_path))['language'] == 'en-US'
    assert calls == [{'download': False, 'langs': None, 'write': None}, {'download': True, 'langs': ['en\\-US'], 'write': True}]


def test_fetch_captions_downloads_nothing_without_a_usable_track(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr('ytdlp_opts.extract_with_bot_check_retry', lambda *a, **k: calls.append(k) or ({}, 'x'))
    info = {'subtitles': {}, 'automatic_captions': {'en': track(translated=True), 'es-orig': track('es')}}
    assert captions.fetch_captions(VIDEO_INFO['url'], 'en', work_dir=str(tmp_path), info=info) is None
    assert calls == []


def test_fetch_captions_reads_the_written_track(monkeypatch, tmp_path, json3):
    written = tmp_path / 'Captioned.en.json3'
    written.write_text(json.dumps(json3), encoding='utf-8')
    info = {'id': 'abc123def45', 'title': 'Captioned', 'channel': 'Chan', 'uploader': 'Up', 'duration': 196.4,
            'upload_date': '20261017', 'subtitles': {}, 'automatic_captions': {'en': track(), 'en-orig': track()},
            'requested_subtitles': {'en': {'ext': 'json3', 'filepath': str(written)}}}
    seen = {}

    def fake_extract(opts, url, proxied, download=True, attempts=None):
        seen.update(opts=opts, url=url, proxied=proxied, download=download)
        return info, str(tmp_path / 'Captioned.mp4')

    monkeypatch.setattr('ytdlp_opts.extract_with_bot_check_retry', fake_extract)
    result = captions.fetch_captions(VIDEO_INFO['url'], 'en', work_dir=str(tmp_path))
    assert seen['download'] is True and seen['opts']['skip_download'] is True and seen['proxied'] is False
    assert result['language'] == 'en' and result['kind'] == 'auto'
    assert result['duration_seconds'] == 196
    assert (result['title'], result['channel_title'], result['upload_date']) == ('Captioned', 'Chan', '10-17-2026')
    assert result['segments'] == cues_to_segments(dedupe_rolling(json3_to_cues(json3)),
                                                  dict(VIDEO_INFO, title='Captioned'))


def test_fetch_captions_returns_none_on_errors(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise RuntimeError('HTTP Error 429')

    monkeypatch.setattr('ytdlp_opts.extract_with_bot_check_retry', boom)
    assert captions.fetch_captions(VIDEO_INFO['url'], work_dir=str(tmp_path)) is None

    monkeypatch.setattr('ytdlp_opts.extract_with_bot_check_retry',
                        lambda *a, **k: ({'subtitles': {}, 'automatic_captions': {}}, 'x'))
    assert captions.fetch_captions(VIDEO_INFO['url'], work_dir=str(tmp_path)) is None

    monkeypatch.setattr('ytdlp_opts.extract_with_bot_check_retry',
                        lambda *a, **k: ({'subtitles': {'en': track()}, 'requested_subtitles': {}}, 'x'))
    assert captions.fetch_captions(VIDEO_INFO['url'], work_dir=str(tmp_path)) is None, 'listed but not written'


# Review rework (finding 2): chained rolling cues compare with the previous cue as
# displayed, not with what was kept of it.
@pytest.mark.parametrize('cues, expected', [
    pytest.param([cue(0, 2, 'a b c d'), cue(2, 4, 'c d e f'), cue(4, 6, 'c d e f g')],
                 [cue(0, 2, 'a b c d'), cue(2, 6, 'e f g')], id='overlap then growth (reviewer repro)'),
    pytest.param([cue(0, 1, 'a b'), cue(1, 2, 'a b c d'), cue(2, 3, 'c d e f')],
                 [cue(0, 2, 'a b c d'), cue(2, 3, 'e f')], id='growth then overlap'),
    pytest.param([cue(0, 1, 'a b c d'), cue(1, 2, 'c d e f'), cue(2, 3, 'e f g h'), cue(3, 4, 'g h i j')],
                 [cue(0, 1, 'a b c d'), cue(1, 2, 'e f'), cue(2, 3, 'g h'), cue(3, 4, 'i j')],
                 id='sliding two-line window'),
    pytest.param([cue(0, 1, 'a b c d'), cue(1, 2, 'c d'), cue(2, 3, 'c d e'), cue(3, 4, 'c d e f')],
                 [cue(0, 4, 'a b c d e f')], id='overlap with nothing new, then growth'),
    pytest.param([cue(0, 1, 'a b c d'), cue(1, 2, 'a b'), cue(2, 3, 'a b c d e')],
                 [cue(0, 3, 'a b c d e')], id='repeat of the opening keeps the longer context'),
    pytest.param([cue(0, 1, 'a b c d'), cue(1, 2, 'c d e f'), cue(2, 3, 'x y'), cue(3, 4, 'x y z')],
                 [cue(0, 1, 'a b c d'), cue(1, 2, 'e f'), cue(2, 4, 'x y z')], id='distinct speech resets context'),
    pytest.param([cue(0, 1, 'a b c d'), cue(1, 2, 'c d e f'), cue(2, 3, 'f g')],
                 [cue(0, 1, 'a b c d'), cue(1, 2, 'e f'), cue(2, 3, 'f g')], id='one shared word stays speech'),
])
def test_dedupe_rolling_chains(cues, expected):
    before = copy.deepcopy(cues)
    out = dedupe_rolling(cues)
    assert out == expected
    assert cues == before
    assert dedupe_rolling(out) == out


def test_dedupe_rolling_chain_keeps_every_new_word_once():
    window = 'w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11'.split()
    cues, t = [], 0.0
    for i in range(0, len(window) - 3, 2):  # 4-word window sliding 2 words, then growing by one
        cues.append(cue(t, t + 1, ' '.join(window[i:i + 4])))
        cues.append(cue(t + 1, t + 2, ' '.join(window[i:i + 5])))
        t += 2
    joined = ' '.join(c['text'] for c in dedupe_rolling(cues))
    assert joined == ' '.join(window)


# Captions fail-fast hotfix: CAPTIONS_ATTEMPTS / CAPTIONS_ENABLED
@pytest.mark.parametrize('value, expected', [(None, 1), ('3', 3), ('1', 1), ('0', 1), ('-2', 1), ('garbage', 1), ('', 1)])
def test_captions_attempts(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv('CAPTIONS_ATTEMPTS', raising=False)
    else:
        monkeypatch.setenv('CAPTIONS_ATTEMPTS', value)
    assert captions.captions_attempts() == expected


@pytest.mark.parametrize('value, expected', [(None, True), ('true', True), ('1', True), ('yes', True),
                                             ('false', False), ('0', False), ('no', False), ('off', False),
                                             ('FALSE', False), (' Off ', False)])
def test_captions_enabled(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv('CAPTIONS_ENABLED', raising=False)
    else:
        monkeypatch.setenv('CAPTIONS_ENABLED', value)
    assert captions.captions_enabled() is expected


@pytest.mark.parametrize('env, expected', [(None, 1), ('2', 2)])
def test_fetch_captions_passes_the_captions_budget_on_both_calls(monkeypatch, tmp_path, json3, env, expected):
    if env is None:
        monkeypatch.delenv('CAPTIONS_ATTEMPTS', raising=False)
    else:
        monkeypatch.setenv('CAPTIONS_ATTEMPTS', env)
    monkeypatch.setenv('YTDLP_BOT_CHECK_ATTEMPTS', '10')
    written = tmp_path / 'Captioned.en.json3'
    written.write_text(json.dumps(json3), encoding='utf-8')
    listing = {'id': 'abc123def45', 'title': 'Captioned', 'subtitles': {}, 'automatic_captions': {'en': track()}}
    downloaded = dict(listing, requested_subtitles={'en': {'ext': 'json3', 'filepath': str(written)}})
    calls = []

    def fake_extract(opts, url, proxied, download=True, attempts=None):
        calls.append((download, attempts))
        return (downloaded if download else listing), 'x'

    monkeypatch.setattr('ytdlp_opts.extract_with_bot_check_retry', fake_extract)
    assert captions.fetch_captions(VIDEO_INFO['url'], 'en', work_dir=str(tmp_path)) is not None
    assert calls == [(False, expected), (True, expected)], 'the probe (info=None) and the download'
    calls.clear()
    assert captions.fetch_captions(VIDEO_INFO['url'], 'en', work_dir=str(tmp_path), info=listing) is not None
    assert calls == [(True, expected)]


def test_audio_download_and_metadata_probe_keep_the_env_budget():
    """Only captions.py passes attempts=: the downloader and the processor's probe call sites are untouched."""
    for name in ('cloud-vm-processor.py', 'yt-dlp-aduio-processor-v1.py'):
        source = open(os.path.join(os.path.dirname(captions.__file__), name), encoding='utf-8').read()
        assert 'attempts=' not in source, name
