"""
Issue #3 P1: the proxy chain (proxy_chain.py) driving extract_with_bot_check_retry.

Offline: a fake YoutubeDL records the proxy each attempt was built with, and each test has
its own state file under tmp_path. Proxy URLs are placeholders.
"""
import json
import os
import subprocess
import sys

import pytest

import proxy_chain
import ytdlp_opts
from proxy_chain import ProxyChain, StaticProvider, WebshareProvider

PHONE = 'http://phone.example:8080'
HOME = 'http://home.example:8888'
POOL = 'http://user-rotate:pw@p.example:80'

E429 = "ERROR: Unable to download video subtitles for 'en': HTTP Error 429: Too Many Requests"
BOT = "ERROR: [youtube] x: Sign in to confirm you're not a bot. Use --cookies-from-browser"
PRIVATE = 'ERROR: [youtube] x: Private video. Sign in if you have been granted access'


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    for var in ('YTDLP_PROXY', 'YTDLP_PROXY_CHAIN', 'YTDLP_BOT_CHECK_ATTEMPTS', 'YTDLP_PROXY_POOL_SIZE',
                'YTDLP_PROXY_COOLDOWN_SECONDS', 'YTDLP_PROXY_SLOT_TTL_SECONDS'):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(ytdlp_opts.time, 'sleep', lambda s: None)
    monkeypatch.setattr(ytdlp_opts, 'PROXY_START', 0)


def fake_ydl(monkeypatch, failure_for):
    """failure_for(proxy) -> an error message to raise, or None to succeed. Returns the proxies used."""
    seen = []

    class FakeYDL:
        def __init__(self, opts):
            self.proxy = opts.get('proxy')
            seen.append(self.proxy)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download):
            message = failure_for(self.proxy)
            if message:
                raise ytdlp_opts.yt_dlp.utils.DownloadError(message)
            return {'id': 'x', 'proxy': self.proxy}

        def prepare_filename(self, info):
            return 'x'

    monkeypatch.setattr(ytdlp_opts.yt_dlp, 'YoutubeDL', FakeYDL)
    return seen


def use_chain(monkeypatch, *providers):
    chain = ProxyChain(providers)
    monkeypatch.setattr(ytdlp_opts.ProxyChain, 'from_env', classmethod(lambda cls, default_url=None: chain))
    return chain


def extract(**kwargs):
    return ytdlp_opts.extract_with_bot_check_retry({'proxy': PHONE}, 'u', proxied=True, **kwargs)


def state(tmp_path):
    with open(tmp_path / proxy_chain.STATE_FILE) as f:
        return json.load(f)['providers']


# 1. ordering
def test_a_429_on_the_first_provider_moves_to_the_second(monkeypatch, tmp_path):
    phone, home = StaticProvider('phone', PHONE), StaticProvider('home', HOME)
    use_chain(monkeypatch, phone, home)
    seen = fake_ydl(monkeypatch, lambda p: E429 if p == PHONE else None)

    info, _ = extract()

    assert info['proxy'] == HOME
    assert seen == [PHONE, HOME]
    assert phone.outcomes == ['throttled'] and home.outcomes == ['ok']
    assert state(tmp_path)['phone']['cooldown_until'] > 0
    assert all(entry['slots'] == [] for entry in state(tmp_path).values()), 'slots released'


# 2. cooldown skip
def test_a_flagged_provider_is_skipped_until_its_cooldown_passes(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(proxy_chain.time, 'time', lambda: clock[0])
    phone, home = StaticProvider('phone', PHONE), StaticProvider('home', HOME)
    use_chain(monkeypatch, phone, home)
    flagged = [True]
    seen = fake_ydl(monkeypatch, lambda p: BOT if p == PHONE and flagged[0] else None)

    extract()
    assert phone.outcomes == ['bot_check']
    flagged[0] = False
    seen.clear()

    clock[0] += 299
    assert extract()[0]['proxy'] == HOME
    assert seen == [HOME], 'phone not touched while cooling down'

    clock[0] += 2
    assert extract()[0]['proxy'] == PHONE
    assert seen == [HOME, PHONE]


# 3. concurrency cap
def test_a_provider_at_its_cap_is_passed_over_and_a_dead_slot_is_reclaimed(monkeypatch, tmp_path):
    phone, home = StaticProvider('phone', PHONE, max_concurrent=1), StaticProvider('home', HOME)
    chain = use_chain(monkeypatch, phone, home)
    seen = fake_ydl(monkeypatch, lambda p: None)

    held = chain.acquire(phone)  # this (live) process holds phone's only slot
    assert held
    assert extract()[0]['proxy'] == HOME
    chain.release(phone, held)

    dead = subprocess.Popen([sys.executable, '-c', 'pass'])
    dead.wait()
    stale = {'providers': {'phone': {'slots': [{'id': 'old', 'pid': dead.pid, 'acquired_at': proxy_chain.time.time()}]}}}
    (tmp_path / proxy_chain.STATE_FILE).write_text(json.dumps(stale))
    seen.clear()
    assert extract()[0]['proxy'] == PHONE, 'slot of a dead pid reclaimed'


def test_a_slot_older_than_the_ttl_is_reclaimed(monkeypatch, tmp_path):
    phone = StaticProvider('phone', PHONE, max_concurrent=1)
    use_chain(monkeypatch, phone, StaticProvider('home', HOME))
    fake_ydl(monkeypatch, lambda p: None)
    old = proxy_chain.time.time() - proxy_chain.DEFAULT_SLOT_TTL_SECONDS - 1
    (tmp_path / proxy_chain.STATE_FILE).write_text(json.dumps(
        {'providers': {'phone': {'slots': [{'id': 'old', 'pid': os.getpid(), 'acquired_at': old}]}}}))
    assert extract()[0]['proxy'] == PHONE


# 4. pinning
def test_each_webshare_attempt_pins_one_url_and_walks_the_pool(monkeypatch):
    monkeypatch.setenv('YTDLP_PROXY_POOL_SIZE', '10')
    pool = WebshareProvider('pool', POOL, attempts=3, pool_size=3)
    use_chain(monkeypatch, pool)
    seen = fake_ydl(monkeypatch, lambda p: E429)

    with pytest.raises(ytdlp_opts.yt_dlp.utils.DownloadError):
        extract()

    # one YoutubeDL per attempt, each built with that attempt's pinned URL; the entry's pool_size wins
    assert seen == ['http://user-1:pw@p.example:80', 'http://user-2:pw@p.example:80', 'http://user-3:pw@p.example:80']
    assert pool.outcomes == ['throttled'] * 3


def test_a_static_provider_keeps_its_one_url_across_attempts(monkeypatch):
    use_chain(monkeypatch, StaticProvider('phone', PHONE, attempts=2))
    seen = fake_ydl(monkeypatch, lambda p: E429)
    with pytest.raises(ytdlp_opts.yt_dlp.utils.DownloadError):
        extract()
    assert seen == [PHONE, PHONE]


# 6. captions budget
def test_the_attempts_cap_is_total_across_the_chain_and_cools_nothing(monkeypatch, tmp_path):
    phone, home = StaticProvider('phone', PHONE), StaticProvider('home', HOME)
    chain = use_chain(monkeypatch, phone, home)
    seen = fake_ydl(monkeypatch, lambda p: E429)

    with pytest.raises(ytdlp_opts.yt_dlp.utils.DownloadError):
        extract(attempts=1)
    assert seen == [PHONE]
    assert 'cooldown_until' not in state(tmp_path)['phone'], 'ending on the captions cap cools nothing'

    chain.cool_down(phone)
    seen.clear()
    with pytest.raises(ytdlp_opts.yt_dlp.utils.DownloadError):
        extract(attempts=1)
    assert seen == [HOME], 'the one attempt lands on the first provider not cooling down'


def test_a_captions_429_does_not_cool_the_only_webshare_provider(monkeypatch, tmp_path):
    monkeypatch.setenv('YTDLP_PROXY', POOL)
    seen = fake_ydl(monkeypatch, lambda p: E429)
    with pytest.raises(ytdlp_opts.yt_dlp.utils.DownloadError):
        ytdlp_opts.extract_with_bot_check_retry({'proxy': POOL}, 'u', proxied=True, attempts=1)
    assert len(seen) == 1
    assert 'cooldown_until' not in state(tmp_path)['webshare']


# all cooling down, non-retryable, success clears
def test_when_every_provider_is_cooling_the_first_is_tried_once_anyway(monkeypatch):
    phone, home = StaticProvider('phone', PHONE, attempts=3), StaticProvider('home', HOME)
    chain = use_chain(monkeypatch, phone, home)
    chain.cool_down(phone)
    chain.cool_down(home)
    seen = fake_ydl(monkeypatch, lambda p: None)
    assert extract()[0]['proxy'] == PHONE
    assert seen == [PHONE]
    assert chain.acquire(phone), 'a success clears the cooldown'


def test_a_non_retryable_failure_raises_at_once_without_cooldown(monkeypatch, tmp_path):
    phone = StaticProvider('phone', PHONE, attempts=3)
    use_chain(monkeypatch, phone, StaticProvider('home', HOME))
    seen = fake_ydl(monkeypatch, lambda p: PRIVATE)
    with pytest.raises(ytdlp_opts.yt_dlp.utils.DownloadError, match='Private video'):
        extract()
    assert seen == [PHONE]
    assert phone.outcomes == ['error']
    assert 'cooldown_until' not in state(tmp_path)['phone']


def test_the_chain_never_logs_a_url_or_credential(monkeypatch, capsys):
    use_chain(monkeypatch, StaticProvider('phone', PHONE), WebshareProvider('pool', POOL, attempts=2, pool_size=3))
    fake_ydl(monkeypatch, lambda p: E429)
    with pytest.raises(ytdlp_opts.yt_dlp.utils.DownloadError):
        extract()
    out = capsys.readouterr().out
    assert 'phone' in out and 'pool' in out
    for secret in ('phone.example', 'p.example', 'pw', 'user-'):
        assert secret not in out


# 7. state shared through the file
def test_two_chains_share_cooldown_through_the_state_file(tmp_path):
    parent = ProxyChain([StaticProvider('phone', PHONE)])
    child = ProxyChain([StaticProvider('phone', PHONE)])
    parent.cool_down(parent.providers[0])
    assert child.acquire(child.providers[0]) is None


def test_a_child_process_cooldown_reaches_the_parent(tmp_path):
    """The audio download runs as a subprocess; what it learns must outlive it."""
    script = ("import proxy_chain as pc; c = pc.ProxyChain([pc.StaticProvider('phone', 'http://phone.example:8080')]); "
              "c.cool_down(c.providers[0])")
    env = dict(os.environ, TMPDIR=str(tmp_path), PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    subprocess.run([sys.executable, '-c', script], env=env, check=True, capture_output=True)
    parent = ProxyChain([StaticProvider('phone', PHONE)])
    assert parent.acquire(parent.providers[0]) is None


@pytest.mark.parametrize('content', [None, '{not json', '[]', '{"providers": 3}'])
def test_a_missing_or_corrupt_state_file_starts_fresh(tmp_path, content):
    if content is not None:
        (tmp_path / proxy_chain.STATE_FILE).write_text(content)
    chain = ProxyChain([StaticProvider('phone', PHONE)])
    slot = chain.acquire(chain.providers[0])
    assert slot
    assert state(tmp_path)['phone']['slots'][0]['id'] == slot


# 8. config
def test_from_env_parses_the_chain(monkeypatch):
    monkeypatch.setenv('YTDLP_PROXY_CHAIN', json.dumps([
        {'type': 'static', 'name': 'phone', 'url': PHONE, 'attempts': 1, 'max_concurrent': 4, 'rotate_url': 'https://rotate.example/x'},
        {'type': 'webshare', 'name': 'pool', 'url': POOL, 'pool_size': 10, 'attempts': 4, 'max_concurrent': 6},
    ]))
    phone, pool = ProxyChain.from_env().providers
    assert (type(phone), phone.name, phone.url, phone.attempts, phone.max_concurrent, phone.rotate_url) == (
        StaticProvider, 'phone', PHONE, 1, 4, 'https://rotate.example/x')
    assert (type(pool), pool.attempts, pool.max_concurrent, pool.pool_size) == (WebshareProvider, 4, 6, 10)
    assert phone.rotate() is False


def test_from_env_without_a_chain_is_today_or_nothing(monkeypatch):
    assert len(ProxyChain.from_env()) == 0
    monkeypatch.setenv('YTDLP_PROXY', POOL)
    monkeypatch.setenv('YTDLP_BOT_CHECK_ATTEMPTS', '4')
    [only] = ProxyChain.from_env().providers
    assert (type(only), only.name, only.url, only.attempts, only.max_concurrent) == (WebshareProvider, 'webshare', POOL, 4, 0)


@pytest.mark.parametrize('chain, named', [
    ('[{"type": "static", "name": "phone", "url": "http://user:pw@phone.example:8080"', None),
    ('[{"type": "socks-ish", "name": "phone", "url": "http://user:pw@phone.example:8080"}]', "'phone'"),
    ('[{"type": "static", "name": "phone"}]', "'phone'"),
    ('[{"type": "static", "name": "phone", "url": "http://user:pw@phone.example:8080", "attempts": 0}]', "'phone'"),
    ('[{"type": "static", "url": "http://user:pw@phone.example:8080"}]', '#1'),
    ('[{"type": "static", "name": "a", "url": "http://user:pw@phone.example:8080"},'
     ' {"type": "static", "name": "a", "url": "http://user:pw@phone.example:8080"}]', None),
    ('{"type": "static"}', None),
    ('[]', None),
])
def test_a_bad_chain_raises_naming_the_entry_never_its_url(monkeypatch, chain, named):
    monkeypatch.setenv('YTDLP_PROXY_CHAIN', chain)
    with pytest.raises(ValueError) as raised:
        ProxyChain.from_env()
    message = str(raised.value)
    assert message.startswith('YTDLP_PROXY_CHAIN')
    if named:
        assert named in message
    for secret in ('phone.example', 'pw', 'user:'):
        assert secret not in message
    assert raised.value.__cause__ is None


# build_ydl_opts with a chain
def test_build_ydl_opts_marks_a_chain_as_proxied_and_logs_names_and_host_only(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv('YTDLP_PROXY_CHAIN', json.dumps([
        {'type': 'static', 'name': 'phone', 'url': 'http://user:pw@phone.example:8080'},
        {'type': 'webshare', 'name': 'pool', 'url': POOL}]))
    cookies = tmp_path / 'cookies.txt'
    cookies.write_text('# Netscape HTTP Cookie File\n')
    monkeypatch.setenv('YOUTUBE_COOKIES_FILE', str(cookies))
    opts = ytdlp_opts.build_ydl_opts(str(tmp_path), audio=True)
    assert opts['proxy'] == 'http://user:pw@phone.example:8080'
    assert 'cookiefile' not in opts
    out = capsys.readouterr().out
    assert 'phone, pool' in out and 'phone.example:8080' in out
    assert 'pw' not in out and 'user' not in out


def test_the_chain_is_used_end_to_end_from_the_env(monkeypatch):
    monkeypatch.setenv('YTDLP_PROXY_CHAIN', json.dumps([
        {'type': 'static', 'name': 'phone', 'url': PHONE},
        {'type': 'webshare', 'name': 'pool', 'url': POOL, 'pool_size': 3, 'attempts': 2}]))
    seen = fake_ydl(monkeypatch, lambda p: BOT if p == PHONE else None)
    opts = ytdlp_opts.build_ydl_opts('/nonexistent', audio=False)
    info, _ = ytdlp_opts.extract_with_bot_check_retry(opts, 'u', proxied=bool(opts.get('proxy')))
    assert seen == [PHONE, 'http://user-1:pw@p.example:80']
