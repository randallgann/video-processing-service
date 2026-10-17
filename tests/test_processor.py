import json
import logging
import py_compile

import pytest

from conftest import REPO_ROOT

URL = 'https://www.youtube.com/watch?v=abc123def45'
INFO = {'id': 'abc123def45', 'title': 'Real Title', 'channel': 'The Channel', 'duration': 754.6,
        'upload_date': '20261017'}
CAPTIONS = {'segments': [{'text': 'Spoken Words: hi', 'metadata': {}}], 'language': 'en', 'kind': 'auto',
            'duration_seconds': 755, 'title': 'Real Title', 'channel_title': 'The Channel',
            'upload_date': '10-17-2026'}


class FakeMessage:
    def __init__(self, payload, message_id='msg-1'):
        self.message_id = message_id
        self.data = json.dumps(payload).encode('utf-8')
        self.acks = 0

    def ack(self):
        self.acks += 1


def request(**overrides):
    payload = {'jobId': 'job-1', 'video': {'id': 'db-video-uuid', 'url': URL, 'title': 't', 'duration': 754},
               'correlation': {'video_id': 'db-video-uuid', 'job_id': 'job-1'}, 'model_type': 'openai'}
    payload.update(overrides)
    return payload


@pytest.fixture
def worker(processor, monkeypatch, tmp_path):
    """process_message with every external collaborator replaced; records what was called."""
    calls = {'completions': [], 'progress': [], 'fetch_captions': [], 'download': 0, 'transcribe': 0,
             'probe': 0, 'uploads': []}
    state = {'info': dict(INFO), 'captions': dict(CAPTIONS), 'download_error': None, 'audio_seconds': 754.2}

    def probe_video(url, work_dir):
        calls['probe'] += 1
        return state['info']

    def fetch_captions(url, lang='en', *, work_dir, info=None):
        calls['fetch_captions'].append(lang)
        calls['fetch_info'] = info
        return state['captions']

    def download_video(url, out_dir):
        calls['download'] += 1
        if state['download_error']:
            raise Exception(state['download_error'])
        audio = tmp_path / 'a.mp3'
        audio.write_bytes(b'')
        desc = tmp_path / 'a.txt'
        desc.write_text('Upload Date: 01-02-2020\nTitle: From Description\n\n', encoding='utf-8')
        return str(audio), str(desc)

    def transcribe(chunks, model_type=None, model_version=None):
        calls['transcribe'] += 1
        return [{'chunk_id': 0, 'start_time': 0, 'duration': 300, 'segments': [
            {'start': 0.0, 'end': 61.0, 'text': 'whisper words'}], 'transcript_text': 'whisper words'}]

    monkeypatch.setattr(processor, 'probe_video', probe_video)
    monkeypatch.setattr(processor, 'fetch_captions', fetch_captions)
    monkeypatch.setattr(processor, 'download_video', download_video)
    monkeypatch.setattr(processor, 'get_audio_duration', lambda path: state['audio_seconds'])
    monkeypatch.setattr(processor, 'split_audio', lambda *a, **k: [{'path': 'c0', 'start_time': 0, 'duration': 300,
                                                                   'id': 0}])
    monkeypatch.setattr(processor, 'upload_chunks', lambda chunks, bucket, mid: chunks)
    monkeypatch.setattr(processor, 'transcribe_chunks_parallel', transcribe)
    monkeypatch.setattr(processor, 'delete_gcs_chunks', lambda *a: 0)
    monkeypatch.setattr(processor, 'upload_results',
                        lambda path, bucket, mid: calls['uploads'].append(json.load(open(path))) or 'signed-url')
    monkeypatch.setattr(processor, 'publish_progress', lambda **kw: calls['progress'].append(kw))
    monkeypatch.setattr(processor, 'publish_completion_result', lambda **kw: calls['completions'].append(kw))

    def run(payload):
        message = FakeMessage(payload)
        processor.process_message(message)
        return message

    return run, calls, state


# 7a / 7e: standard (explicit or absent) with captions available
@pytest.mark.parametrize('payload', [request(quality='standard'), request()], ids=['standard', 'quality absent'])
def test_standard_with_captions_skips_download(worker, payload):
    run, calls, _ = worker
    message = run(payload)
    assert calls['download'] == 0 and calls['transcribe'] == 0
    assert calls['fetch_captions'] == ['en']
    assert calls['fetch_info'] == INFO, 'the probe info is reused: pick from it, download one track'
    assert message.acks == 1
    [completion] = calls['completions']
    assert completion['status'] == 'completed'
    assert completion['transcript_source'] == 'captions'
    assert completion['language'] == 'en'
    assert completion['duration_seconds'] == 755
    assert (completion['title'], completion['channel_title']) == ('Real Title', 'The Channel')
    assert completion['transcript_location'].endswith('/msg-1/results/transcript_abc123def45.json')
    assert completion['db_video_id'] == 'db-video-uuid'
    assert completion['correlation'] == {'video_id': 'db-video-uuid', 'job_id': 'job-1'}
    assert calls['uploads'] == [CAPTIONS['segments']]
    assert [p['current_stage'] for p in calls['progress'] if p['current_stage'] == 'captions'] == ['captions'] * 2
    assert {p['progress_percent'] for p in calls['progress'] if p['current_stage'] == 'captions'} == {50, 90}


def test_caption_track_language_wins_and_request_language_is_used_for_lookup(worker):
    run, calls, state = worker
    state['captions'] = dict(CAPTIONS, language='es')
    run(request(language='es'))
    assert calls['fetch_captions'] == ['es']
    assert calls['completions'][0]['language'] == 'es'


# 7b: no captions -> Whisper
def test_standard_without_captions_falls_back_to_whisper(worker):
    run, calls, state = worker
    state['captions'] = None
    message = run(request(quality='standard', language='de'))
    assert calls['fetch_captions'] == ['de']
    assert calls['download'] == 1 and calls['transcribe'] == 1
    assert message.acks == 1
    [completion] = calls['completions']
    assert completion['status'] == 'completed'
    assert completion['transcript_source'] == 'whisper'
    assert completion['language'] == 'de', 'request language when captions were not used'
    assert completion['duration_seconds'] == 755, 'probe duration, rounded'
    assert (completion['title'], completion['channel_title']) == ('Real Title', 'The Channel')
    [uploaded] = calls['uploads']
    assert uploaded[0]['metadata']['title'] == 'Real Title'
    assert uploaded[0]['metadata']['upload_date'] == '10-17-2026'


# 7c: whisper never tries captions
def test_whisper_quality_never_fetches_captions(worker):
    run, calls, _ = worker
    run(request(quality='whisper'))
    assert calls['fetch_captions'] == []
    assert calls['download'] == 1 and calls['transcribe'] == 1
    assert calls['completions'][0]['transcript_source'] == 'whisper'
    assert calls['completions'][0]['language'] == 'en'


def test_unknown_quality_is_standard(worker, caplog):
    run, calls, _ = worker
    with caplog.at_level(logging.WARNING):
        run(request(quality='premium'))
    assert calls['fetch_captions'] == ['en']
    assert "Unknown quality 'premium'" in caplog.text


# 7d: over the cap -> failed, acked, nothing downloaded
def test_over_duration_cap_fails_and_acks(worker, processor, monkeypatch):
    run, calls, state = worker
    monkeypatch.setattr(processor, 'MAX_VIDEO_DURATION_SECONDS', 600)
    message = run(request(quality='standard'))
    assert calls['download'] == 0 and calls['fetch_captions'] == [] and calls['transcribe'] == 0
    assert message.acks == 1
    [completion] = calls['completions']
    assert completion['status'] == 'failed'
    assert completion['error'] == 'Video is 755s, over the 600s limit'
    assert 'transcript_source' not in completion


@pytest.mark.parametrize('quality', ['whisper', 'standard'])
def test_probe_without_duration_caps_on_audio_before_spend(worker, processor, monkeypatch, quality):
    """Review finding 1: a probe that knows the video but not its duration must not bypass the cap."""
    run, calls, state = worker
    state['info'] = dict(INFO, duration=None)
    state['captions'] = None
    state['audio_seconds'] = 15000.0
    message = run(request(quality=quality))
    assert calls['download'] == 1
    assert calls['transcribe'] == 0, 'no Replicate spend for an over-cap video'
    assert calls['uploads'] == []
    assert message.acks == 1
    [completion] = calls['completions']
    assert completion['status'] == 'failed'
    assert completion['error'] == 'Video is 15000s, over the 14400s limit'


def test_probe_without_duration_caps_caption_duration(worker):
    run, calls, state = worker
    state['info'] = dict(INFO, duration=None)
    state['captions'] = dict(CAPTIONS, duration_seconds=15000)
    message = run(request(quality='standard'))
    assert calls['uploads'] == [] and calls['download'] == 0
    assert message.acks == 1
    [completion] = calls['completions']
    assert completion['status'] == 'failed'
    assert completion['error'] == 'Video is 15000s, over the 14400s limit'


def test_probe_without_duration_under_cap_keeps_metadata(worker):
    run, calls, state = worker
    state['info'] = dict(INFO, duration=None)
    state['captions'] = None
    state['audio_seconds'] = 700.9
    run(request(quality='standard'))
    completion = calls['completions'][0]
    assert completion['status'] == 'completed'
    assert completion['duration_seconds'] == 700
    assert (completion['title'], completion['channel_title']) == ('Real Title', 'The Channel')
    assert calls['uploads'][0][0]['metadata']['upload_date'] == '10-17-2026'

    calls['completions'].clear()
    state['captions'] = dict(CAPTIONS, duration_seconds=755)
    run(request(quality='standard'))
    assert calls['completions'][0]['transcript_source'] == 'captions'
    assert calls['completions'][0]['duration_seconds'] == 755


def test_default_cap_is_four_hours(processor):
    assert processor.MAX_VIDEO_DURATION_SECONDS == 14400


def test_probe_failure_goes_to_whisper_and_caps_on_audio(worker, processor, monkeypatch):
    run, calls, state = worker
    state['info'] = None
    state['audio_seconds'] = 700.9
    run(request(quality='standard'))
    assert calls['fetch_captions'] == [] and calls['download'] == 1
    completion = calls['completions'][0]
    assert completion['transcript_source'] == 'whisper'
    assert completion['duration_seconds'] == 700
    assert completion['title'] == 'From Description' and completion['channel_title'] is None
    assert calls['uploads'][0][0]['metadata']['upload_date'] == '01-02-2020'

    calls['completions'].clear()
    monkeypatch.setattr(processor, 'MAX_VIDEO_DURATION_SECONDS', 600)
    message = run(request(quality='whisper'))
    assert calls['transcribe'] == 1, 'Replicate is never called for the over-cap video'
    assert message.acks == 1
    assert calls['completions'][0]['error'] == 'Video is 701s, over the 600s limit'


def test_bot_check_download_failure_is_acked(worker):
    run, calls, state = worker
    state['captions'] = None
    state['download_error'] = "Failed to download video: ERROR: Sign in to confirm you're not a bot"
    message = run(request())
    assert message.acks == 1
    assert calls['completions'][0]['status'] == 'failed'


def test_transient_download_failure_is_still_redelivered(worker):
    run, calls, state = worker
    state['captions'] = None
    state['download_error'] = 'Failed to download video: ERROR: timed out'
    message = run(request())
    assert message.acks == 0
    assert calls['completions'][0]['status'] == 'failed'


# Completion message shape: T2 fields only when not None
def test_completion_message_adds_t2_fields_only_when_set(processor, monkeypatch):
    published = []

    class FakeFuture:
        def result(self):
            return 'id'

    class FakePublisher:
        def topic_path(self, project, topic):
            return f'projects/{project}/topics/{topic}'

        def publish(self, topic, data):
            published.append(json.loads(data))
            return FakeFuture()

    monkeypatch.setattr(processor, 'RESULTS_TOPIC_ID', 'results')
    monkeypatch.setattr(processor.pubsub_v1, 'PublisherClient', FakePublisher)
    processor.publish_completion_result(message_id='m', video_url=URL, status='failed', error='x')
    processor.publish_completion_result(message_id='m', video_url=URL, status='completed', transcript_location='gs://b/p',
                                        transcript_source='captions', language='en', duration_seconds=0,
                                        title='T', channel_title=None)
    legacy, rich = published
    assert set(legacy) == {'message_id', 'video_id', 'video_url', 'status', 'timestamp', 'error'}
    assert {k: rich[k] for k in ('transcript_source', 'language', 'duration_seconds', 'title')} == {
        'transcript_source': 'captions', 'language': 'en', 'duration_seconds': 0, 'title': 'T'}
    assert 'channel_title' not in rich


# Throttle visibility
def test_throttle_is_counted_and_logged(processor, monkeypatch, caplog):
    attempts = iter([Exception('Request was throttled. Your rate limit resets in ~1s'), 'ok'])

    def run(*args, **kwargs):
        value = next(attempts)
        if isinstance(value, Exception):
            raise value
        return {'transcription': '1\n00:00:00,000 --> 00:00:01,000\nhi\n'}

    monkeypatch.setattr(processor.replicate, 'run', run, raising=False)
    monkeypatch.setattr(processor.time, 'sleep', lambda s: None)
    monkeypatch.setattr(processor, 'OPENAI_WHISPER', 'openai/whisper:v1')
    before = processor.replicate_throttled_total
    with caplog.at_level(logging.WARNING):
        result = processor.transcribe_chunk_replicate({'id': 0, 'gcs_url': 'u', 'start_time': 0, 'duration': 1})
    assert result['segments'] == [{'start': 0.0, 'end': 1.0, 'text': 'hi'}]
    assert processor.replicate_throttled_total == before + 1
    assert f'replicate_throttled_total={before + 1}' in caplog.text
    assert 'below $5 credit' in caplog.text


# 8. SIGTERM
def test_sigterm_handler_cancels_once_and_flips_readiness(processor, monkeypatch):
    class FakeFuture:
        cancels = 0

        def cancel(self):
            self.cancels += 1
            return True

    future = FakeFuture()
    monkeypatch.setattr(processor, 'service_ready', True)
    handler = processor.make_sigterm_handler(future)
    handler(15, None)
    assert future.cancels == 1
    assert processor.service_ready is False
    monkeypatch.setattr(processor, 'service_ready', True)
    handler(15, None)
    assert future.cancels == 1, 'second SIGTERM is a no-op'
    assert processor.service_ready is True


def test_main_subscribes_with_callback_draining_and_registers_sigterm(processor, monkeypatch):
    seen = {}

    class FakeFuture:
        def result(self):
            seen['handler'](15, None)  # SIGTERM arrives while main waits
            return True

        def cancel(self):
            seen['cancelled'] = True
            return True

    class FakeSubscriber:
        def subscription_path(self, project, sub):
            return 'sub-path'

        def subscribe(self, path, **kwargs):
            seen['kwargs'] = kwargs
            return FakeFuture()

    monkeypatch.setattr(processor, 'check_environment', lambda: True)
    monkeypatch.setattr(processor, 'start_health_server', lambda: None)
    monkeypatch.setattr(processor.pubsub_v1, 'SubscriberClient', FakeSubscriber)
    monkeypatch.setattr(processor.signal, 'signal', lambda sig, handler: seen.update(sig=sig, handler=handler))
    with pytest.raises(SystemExit) as exit_info:
        processor.main()
    assert exit_info.value.code == 0
    assert seen['kwargs']['await_callbacks_on_shutdown'] is True
    assert seen['sig'] == processor.signal.SIGTERM
    assert seen['cancelled'] is True


# 9. Everything the image runs compiles
@pytest.mark.parametrize('name', ['cloud-vm-processor.py', 'yt-dlp-aduio-processor-v1.py', 'ytdlp_opts.py',
                                  'captions.py'])
def test_compiles(name, tmp_path):
    py_compile.compile(f'{REPO_ROOT}/{name}', doraise=True, cfile=str(tmp_path / 'out.pyc'))


def test_dockerfile_copies_new_modules():
    with open(f'{REPO_ROOT}/Dockerfile.optimized', encoding='utf-8') as f:
        dockerfile = f.read()
    for name in ('ytdlp_opts.py', 'captions.py', 'cloud-vm-processor.py', 'yt-dlp-aduio-processor-v1.py'):
        assert f'COPY {name} /app/' in dockerfile
