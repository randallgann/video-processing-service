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
def test_when_every_provider_is_cooling_the_first_is_tried_with_its_full_budget(monkeypatch):
    """R8: cooldown only orders/skips; the last resort keeps the provider's attempts."""
    phone, home = StaticProvider('phone', PHONE, attempts=3), StaticProvider('home', HOME)
    chain = use_chain(monkeypatch, phone, home)
    chain.cool_down(phone)
    chain.cool_down(home)
    failures = [2]

    def flaky(p):
        if failures[0]:
            failures[0] -= 1
            return E429
        return None

    seen = fake_ydl(monkeypatch, flaky)
    assert extract()[0]['proxy'] == PHONE
    assert seen == [PHONE, PHONE, PHONE], 'all three of phone\'s attempts, never home'
    assert chain.acquire(phone), 'a success clears the cooldown'


def test_a_cooled_single_webshare_provider_keeps_todays_download_budget(monkeypatch, tmp_path):
    """R8 back-compat: YTDLP_PROXY only, provider cooled, a download call still makes 10 attempts."""
    monkeypatch.setenv('YTDLP_PROXY', POOL)
    monkeypatch.setenv('YTDLP_PROXY_POOL_SIZE', '10')
    seen = fake_ydl(monkeypatch, lambda p: E429)
    with pytest.raises(ytdlp_opts.yt_dlp.utils.DownloadError):
        ytdlp_opts.extract_with_bot_check_retry({'proxy': POOL}, 'u', proxied=True)
    assert len(seen) == 10
    assert state(tmp_path)['webshare']['cooldown_until'] > 0, 'exhausted -> cooling down'

    seen.clear()
    with pytest.raises(ytdlp_opts.yt_dlp.utils.DownloadError):
        ytdlp_opts.extract_with_bot_check_retry({'proxy': POOL}, 'u', proxied=True)
    assert seen == [f'http://user-{n}:pw@p.example:80' for n in range(1, 11)], 'same 10 pinned URLs as before the chain'

    seen.clear()
    with pytest.raises(ytdlp_opts.yt_dlp.utils.DownloadError):
        ytdlp_opts.extract_with_bot_check_retry({'proxy': POOL}, 'u', proxied=True, attempts=1)
    assert len(seen) == 1, 'the captions cap still applies to the last resort'


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


# R9: the concurrency cap is never bypassed
CAP_ENV = {'YTDLP_PROXY_CAPACITY_WAIT_SECONDS': '10', 'YTDLP_PROXY_CAPACITY_POLL_SECONDS': '2'}


def fake_clock(monkeypatch, on_sleep=None):
    """time.monotonic advanced only by time.sleep; on_sleep(n) runs on the n-th sleep."""
    now = [100.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds
        if on_sleep:
            on_sleep(len(sleeps))

    monkeypatch.setattr(ytdlp_opts.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(ytdlp_opts.time, 'sleep', sleep)
    return sleeps


def capped_ydl(monkeypatch, chain, provider):
    """Like fake_ydl, but fails the test if a YoutubeDL is built while the provider is over its cap."""
    built = []

    class FakeYDL:
        def __init__(self, opts):
            with open(chain.state_path) as f:
                slots = json.load(f)['providers'][provider.name]['slots']
            assert len(slots) <= provider.max_concurrent, 'provider concurrency cap exceeded'
            built.append(opts['proxy'])

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download):
            return {'id': 'x'}

        def prepare_filename(self, info):
            return 'x'

    monkeypatch.setattr(ytdlp_opts.yt_dlp, 'YoutubeDL', FakeYDL)
    return built


def test_a_full_provider_is_never_used_past_its_cap_and_the_wait_is_bounded(monkeypatch, tmp_path, processor):
    for key, value in CAP_ENV.items():
        monkeypatch.setenv(key, value)
    phone = StaticProvider('phone', PHONE, max_concurrent=1)
    chain = use_chain(monkeypatch, phone)
    built = capped_ydl(monkeypatch, chain, phone)
    sleeps = fake_clock(monkeypatch)
    held = chain.acquire(phone)

    with pytest.raises(proxy_chain.ProxyCapacityError) as raised:
        extract()

    assert built == [], 'no YoutubeDL while the only slot is held'
    assert [s['id'] for s in state(tmp_path)['phone']['slots']] == [held], 'no second slot was ever taken'
    assert sum(sleeps) == 10 and all(s <= 2 for s in sleeps)
    assert str(raised.value) == 'No proxy slot free within 10s: every provider is at its concurrency cap (phone)'
    assert processor.classify_failure(str(raised.value)) == 'retry', 'the job is redelivered, not failed'


def test_a_slot_released_during_the_wait_is_taken(monkeypatch, tmp_path):
    for key, value in CAP_ENV.items():
        monkeypatch.setenv(key, value)
    phone = StaticProvider('phone', PHONE, max_concurrent=1)
    chain = use_chain(monkeypatch, phone)
    built = capped_ydl(monkeypatch, chain, phone)
    held = chain.acquire(phone)
    sleeps = fake_clock(monkeypatch, on_sleep=lambda n: n == 2 and chain.release(phone, held))

    info, _ = extract()

    assert info == {'id': 'x'} and built == [PHONE]
    assert len(sleeps) == 2
    assert state(tmp_path)['phone']['slots'] == []


def test_the_last_resort_takes_a_cooling_provider_with_room_before_waiting(monkeypatch):
    phone, home = StaticProvider('phone', PHONE, max_concurrent=1), StaticProvider('home', HOME, attempts=2)
    chain = use_chain(monkeypatch, phone, home)
    chain.acquire(phone)
    chain.cool_down(home)
    sleeps = fake_clock(monkeypatch)
    seen = fake_ydl(monkeypatch, lambda p: None)
    assert extract()[0]['proxy'] == HOME
    assert seen == [HOME] and sleeps == []


def test_without_state_a_capped_provider_is_not_used(tmp_path):
    missing = str(tmp_path / 'no-such-dir' / proxy_chain.STATE_FILE)
    capped = ProxyChain([StaticProvider('phone', PHONE, max_concurrent=1)], state_path=missing)
    uncapped = ProxyChain([StaticProvider('home', HOME)], state_path=missing)
    assert capped.acquire(capped.providers[0]) is None
    assert uncapped.acquire(uncapped.providers[0]) == 'untracked'


# R10: proxy credentials never appear in what is printed or raised
SECRET_URL = 'http://synthetic-user:synthetic-pass@phone.example:8080'
SECRET_POOL = 'http://synthetic-user-rotate:synthetic-pass@p.example:80'
SECRETS = ('synthetic-user', 'synthetic-pass')


def assert_unchained(error):
    """No traceback can print the original (credential-bearing) exception."""
    assert error.__cause__ is None
    assert error.__context__ is None or error.__suppress_context__


def assert_clean(*texts):
    for text in texts:
        for secret in SECRETS:
            assert secret not in text


def test_a_retry_diagnostic_quoting_the_proxy_url_is_redacted(monkeypatch, capsys):
    """The reviewer's repro: a retryable 429 whose text carries the authenticated proxy URL."""
    use_chain(monkeypatch, StaticProvider('phone', SECRET_URL, attempts=2))
    fake_ydl(monkeypatch, lambda p: 'HTTP Error 429 via ' + p)
    with pytest.raises(ytdlp_opts.yt_dlp.utils.DownloadError) as raised:
        extract()
    out, err = capsys.readouterr()
    assert 'http://***@phone.example:8080' in out
    assert_clean(out, err, str(raised.value), repr(raised.value))
    assert_unchained(raised.value)


@pytest.mark.parametrize('failure', [
    PRIVATE + ' (proxy ' + SECRET_URL + ')',                        # non-retryable
    E429 + ' user synthetic-user, password synthetic-pass',          # bare fragments
])
def test_raised_errors_are_redacted(monkeypatch, failure):
    use_chain(monkeypatch, StaticProvider('phone', SECRET_URL))
    fake_ydl(monkeypatch, lambda p: failure)
    with pytest.raises(ytdlp_opts.yt_dlp.utils.DownloadError) as raised:
        extract()
    assert_clean(str(raised.value))


def test_a_pinned_webshare_url_in_an_error_is_redacted(monkeypatch, capsys):
    use_chain(monkeypatch, WebshareProvider('pool', SECRET_POOL, attempts=2, pool_size=3))
    fake_ydl(monkeypatch, lambda p: f'HTTP Error 403 Forbidden via {p} for synthetic-user-2')
    with pytest.raises(ytdlp_opts.yt_dlp.utils.DownloadError) as raised:
        extract()
    out, err = capsys.readouterr()
    assert_clean(out, err, str(raised.value))


def test_other_exceptions_carrying_credentials_are_redacted(monkeypatch):
    use_chain(monkeypatch, StaticProvider('phone', SECRET_URL))

    class Boom:
        def __init__(self, opts):
            raise OSError(f"Unable to connect to proxy {opts['proxy']}")

    monkeypatch.setattr(ytdlp_opts.yt_dlp, 'YoutubeDL', Boom)
    with pytest.raises(RuntimeError) as raised:
        extract()
    assert str(raised.value) == 'Unable to connect to proxy http://***@phone.example:8080'
    assert_unchained(raised.value)


def test_yt_dlp_own_output_goes_through_the_redacting_logger(monkeypatch, capsys):
    """yt-dlp prints its ERROR:/WARNING: lines itself; the processor logs the child's stderr."""
    use_chain(monkeypatch, StaticProvider('phone', SECRET_URL))

    class TalkativeYDL:
        def __init__(self, opts):
            self.log = opts['logger']

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download):
            self.log.debug(f'[youtube] x: Downloading webpage via {SECRET_URL}')
            self.log.warning('proxy synthetic-user is slow')
            self.log.error(f'ERROR: [youtube] x: HTTP Error 429: Too Many Requests ({SECRET_URL})')
            raise ytdlp_opts.yt_dlp.utils.DownloadError('ERROR: [youtube] x: HTTP Error 429: Too Many Requests')

        def prepare_filename(self, info):
            return 'x'

    monkeypatch.setattr(ytdlp_opts.yt_dlp, 'YoutubeDL', TalkativeYDL)
    with pytest.raises(ytdlp_opts.yt_dlp.utils.DownloadError):
        extract()
    out, err = capsys.readouterr()
    assert_clean(out, err)
    assert '[youtube] x: Downloading webpage via http://***@phone.example:8080' in out
    assert 'WARNING: proxy *** is slow' in err
    assert err.splitlines()[-1].startswith('ERROR: [youtube] x: HTTP Error 429'), 'the processor still finds ERROR:'


def test_the_redacting_logger_matches_the_real_yt_dlp_interface(capsys):
    """Offline: a real YoutubeDL routes to_screen, warnings and errors to our logger."""
    real = ytdlp_opts.yt_dlp
    if not isinstance(real.YoutubeDL, type) or real.YoutubeDL is object:
        pytest.skip('yt-dlp not installed')
    secrets = proxy_chain.url_secrets(SECRET_URL)
    with real.YoutubeDL({'logger': ytdlp_opts._RedactingLogger(secrets), 'quiet': False}) as ydl:
        ydl.to_screen(f'[info] via {SECRET_URL}')
        ydl.report_warning(f'slow proxy {SECRET_URL}')
        with pytest.raises(real.utils.DownloadError) as raised:
            ydl.report_error(f'boom via {SECRET_URL}')
    out, err = capsys.readouterr()
    assert '[info] via http://***@phone.example:8080' in out
    assert 'WARNING: slow proxy http://***@phone.example:8080' in err
    assert 'ERROR: boom via http://***@phone.example:8080' in err
    assert_clean(out, err)
    assert 'synthetic-pass' in str(raised.value), 'yt-dlp itself raises the raw text; extract_with_bot_check_retry cleans it'


def test_the_state_file_never_holds_a_credential(monkeypatch, tmp_path):
    chain = use_chain(monkeypatch, StaticProvider('phone', SECRET_URL), WebshareProvider('pool', SECRET_POOL))
    fake_ydl(monkeypatch, lambda p: E429 if 'phone' in p else None)
    extract()
    chain.acquire(chain.providers[1])
    assert_clean((tmp_path / proxy_chain.STATE_FILE).read_text())
