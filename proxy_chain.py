"""
Ordered YouTube egress options for yt-dlp (issue #3, PROXY_CHAIN_PLAN.md).

To yt-dlp every option is a proxy URL. A provider says how to get the URL for one attempt,
how many attempts it is worth and how many jobs may share it at once. The chain tries
providers in order, skips one that is cooling down after it was flagged, and caps each
provider's concurrent use.

Cooldowns and in-use slots live in one JSON file under $TMPDIR, guarded by flock, because
the audio download runs in a child process (yt-dlp-aduio-processor-v1.py): state kept in
memory would die with the child and the parent would never learn the provider was flagged.
A cooldown only orders and skips providers; it never shrinks the last one's attempts.

The concurrency cap is never bypassed (R9): when every provider is at its cap the caller
waits for a slot (YTDLP_PROXY_CAPACITY_WAIT_SECONDS) and then fails with ProxyCapacityError,
which the processor treats as retryable (the job is redelivered).

Config comes from YTDLP_PROXY_CHAIN (JSON list, from the k8s Secret). Without it the chain
is one webshare provider built from YTDLP_PROXY, which is today's behaviour. Provider URLs
carry credentials: only provider names are logged or put in our own messages, and redact()
strips credentials from any other text (yt-dlp errors and output) before it is shown (R10).
"""
import fcntl
import json
import os
import re
import tempfile
import time
import urllib.parse
import uuid

STATE_FILE = 'proxy-chain-state.json'
DEFAULT_COOLDOWN_SECONDS = 300
DEFAULT_SLOT_TTL_SECONDS = 1800
DEFAULT_CAPACITY_WAIT_SECONDS = 60
DEFAULT_CAPACITY_POLL_SECONDS = 2

# scheme://userinfo@ -> scheme://***@ (userinfo cannot contain '/', '@' or whitespace).
_USERINFO = re.compile(r'(?i)\b([a-z][a-z0-9+.\-]*://)[^\s/@]+@')


def url_secrets(url):
    """The credential fragments of a proxy URL that could appear on their own in some text."""
    parsed = urllib.parse.urlparse(url or '')
    found = []
    for part in (parsed.username, parsed.password):
        if part:
            found += [part, urllib.parse.unquote(part)]
    if parsed.username and parsed.username.endswith('-rotate'):
        found.append(parsed.username[:-len('rotate')])  # the stem of every pinned `<user>-<n>`
    return found


def redact(text, secrets=()):
    """Text with every URL's userinfo replaced by *** and any known credential fragment removed."""
    text = _USERINFO.sub(r'\1***@', str(text))
    for secret in sorted({s for s in secrets if s and len(s) >= 4}, key=len, reverse=True):
        text = text.replace(secret, '***')
    return text


class ProxyCapacityError(RuntimeError):
    """Every provider stayed at its concurrency cap for the whole wait. Retryable: redeliver the job."""


class ProxyProvider:
    """One egress option. max_concurrent 0 means no cap."""

    kind = None

    def __init__(self, name, url, attempts=1, max_concurrent=0):
        self.name = name
        self.url = url
        self.attempts = attempts
        self.max_concurrent = max_concurrent
        self.outcomes = []

    def url_for(self, attempt):
        """The URL for one whole attempt (extraction and download on the same exit IP)."""
        return self.url

    def rotate(self):
        """Ask for a fresh IP. No provider can yet (P4 adds the rotate_url call)."""
        return False

    def report(self, outcome):
        """ok | throttled | bot_check | error. The chain applies the cooldown."""
        self.outcomes.append(outcome)


class StaticProvider(ProxyProvider):
    """One URL: a phone, a home box, a static residential IP."""

    kind = 'static'

    def __init__(self, name, url, attempts=1, max_concurrent=0, rotate_url=None):
        super().__init__(name, url, attempts, max_concurrent)
        self.rotate_url = rotate_url


class WebshareProvider(ProxyProvider):
    """Webshare's sticky pool: `<user>-rotate` pinned to `<user>-<n>` per attempt."""

    kind = 'webshare'

    def __init__(self, name, url, attempts=1, max_concurrent=0, pool_size=None):
        super().__init__(name, url, attempts, max_concurrent)
        self.pool_size = pool_size

    def url_for(self, attempt):
        import ytdlp_opts  # the pinning helper (and its PROXY_START) stays where the downloader imports it
        return ytdlp_opts.proxy_for_attempt(self.url, attempt, self.pool_size)


PROVIDER_TYPES = {cls.kind: cls for cls in (StaticProvider, WebshareProvider)}


def _positive_int(entry, key, label, default, minimum=1):
    value = entry.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"YTDLP_PROXY_CHAIN entry {label}: '{key}' must be an integer >= {minimum}")
    return value


def _provider_from_entry(index, entry):
    if not isinstance(entry, dict):
        raise ValueError(f"YTDLP_PROXY_CHAIN entry #{index + 1} is not an object")
    name = entry.get('name')
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"YTDLP_PROXY_CHAIN entry #{index + 1} has no name")
    label = f"'{name}'"
    cls = PROVIDER_TYPES.get(entry.get('type'))
    if cls is None:
        raise ValueError(f"YTDLP_PROXY_CHAIN entry {label} has an unknown type (expected one of {sorted(PROVIDER_TYPES)})")
    url = entry.get('url')
    if not isinstance(url, str) or not url.strip():
        raise ValueError(f"YTDLP_PROXY_CHAIN entry {label} has no url")
    attempts = _positive_int(entry, 'attempts', label, 1)
    max_concurrent = _positive_int(entry, 'max_concurrent', label, 0, minimum=0)
    if cls is WebshareProvider:
        pool_size = _positive_int(entry, 'pool_size', label, None) if 'pool_size' in entry else None
        return WebshareProvider(name, url, attempts, max_concurrent, pool_size=pool_size)
    rotate_url = entry.get('rotate_url')
    if rotate_url is not None and not isinstance(rotate_url, str):
        raise ValueError(f"YTDLP_PROXY_CHAIN entry {label}: 'rotate_url' must be a string")
    return StaticProvider(name, url, attempts, max_concurrent, rotate_url=rotate_url)


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class ProxyChain:
    def __init__(self, providers, state_path=None, cooldown_seconds=None, slot_ttl_seconds=None):
        self.providers = list(providers)
        self.state_path = state_path or os.path.join(os.environ.get('TMPDIR') or tempfile.gettempdir(), STATE_FILE)
        self.cooldown_seconds = cooldown_seconds if cooldown_seconds is not None else int(
            os.environ.get('YTDLP_PROXY_COOLDOWN_SECONDS', DEFAULT_COOLDOWN_SECONDS))
        self.slot_ttl_seconds = slot_ttl_seconds if slot_ttl_seconds is not None else int(
            os.environ.get('YTDLP_PROXY_SLOT_TTL_SECONDS', DEFAULT_SLOT_TTL_SECONDS))

    def __len__(self):
        return len(self.providers)

    def secrets(self):
        found = []
        for provider in self.providers:
            found += url_secrets(provider.url)
            rotate_url = getattr(provider, 'rotate_url', None)
            if rotate_url:
                found += [rotate_url] + url_secrets(rotate_url)
        return found

    @classmethod
    def from_env(cls, default_url=None):
        """
        YTDLP_PROXY_CHAIN when set; otherwise one webshare provider from default_url (the URL the
        caller's options already carry) or YTDLP_PROXY, with YTDLP_BOT_CHECK_ATTEMPTS attempts and
        no concurrency cap, as before the chain; otherwise an empty chain (no proxy).
        Raises ValueError for a bad chain, naming the entry, never its URL.
        """
        raw = os.environ.get('YTDLP_PROXY_CHAIN', '').strip()
        if raw:
            try:
                entries = json.loads(raw)
            except ValueError:
                raise ValueError('YTDLP_PROXY_CHAIN is not valid JSON') from None
            if not isinstance(entries, list) or not entries:
                raise ValueError('YTDLP_PROXY_CHAIN must be a non-empty JSON list')
            providers = [_provider_from_entry(i, e) for i, e in enumerate(entries)]
            names = [p.name for p in providers]
            if len(set(names)) != len(names):
                raise ValueError('YTDLP_PROXY_CHAIN has duplicate provider names')
            return cls(providers)
        url = default_url or os.environ.get('YTDLP_PROXY')
        if not url:
            return cls([])
        attempts = max(1, int(os.environ.get('YTDLP_BOT_CHECK_ATTEMPTS', '10')))
        return cls([WebshareProvider('webshare', url, attempts=attempts)])

    # --- shared state -------------------------------------------------------------------------

    def _update(self, change):
        """Run change(state) under an exclusive lock and write the result. Returns change's value."""
        with open(self.state_path + '.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                with open(self.state_path) as f:
                    state = json.load(f)
                if not isinstance(state, dict) or not isinstance(state.get('providers'), dict):
                    raise ValueError
            except (OSError, ValueError):
                state = {'providers': {}}
            result = change(state)
            tmp = f"{self.state_path}.{os.getpid()}.tmp"
            with open(tmp, 'w') as f:
                json.dump(state, f)
            os.replace(tmp, self.state_path)
            return result

    def _entry(self, state, provider):
        entry = state['providers'].setdefault(provider.name, {})
        if not isinstance(entry.get('slots'), list):
            entry['slots'] = []
        now = time.time()
        entry['slots'] = [
            s for s in entry['slots']
            if isinstance(s, dict) and isinstance(s.get('pid'), int) and isinstance(s.get('acquired_at'), (int, float))
            and now - s['acquired_at'] < self.slot_ttl_seconds and _pid_alive(s['pid'])
        ]
        return entry

    def acquire(self, provider, ignore_cooldown=False):
        """
        A slot id for one call on this provider, or None while it cools down or is at
        max_concurrent. ignore_cooldown=True (the chain's last resort) skips only the cooldown
        check: the cap always holds.
        """
        def change(state):
            entry = self._entry(state, provider)
            if not ignore_cooldown and entry.get('cooldown_until', 0) > time.time():
                return None
            if provider.max_concurrent and len(entry['slots']) >= provider.max_concurrent:
                return None
            slot = uuid.uuid4().hex
            entry['slots'].append({'id': slot, 'pid': os.getpid(), 'acquired_at': time.time()})
            return slot

        try:
            return self._update(change)
        except OSError as e:
            if provider.max_concurrent:
                # Without the state its cap cannot be checked, and the cap is never bypassed.
                print(f"WARNING: proxy chain state unavailable ({type(e).__name__}); not using capped {provider.name}")
                return None
            print(f"WARNING: proxy chain state unavailable ({type(e).__name__}); using {provider.name} without it")
            return 'untracked'

    def release(self, provider, slot):
        def change(state):
            entry = self._entry(state, provider)
            entry['slots'] = [s for s in entry['slots'] if s.get('id') != slot]

        self._safe_update(change)

    def cool_down(self, provider):
        def change(state):
            self._entry(state, provider)['cooldown_until'] = time.time() + self.cooldown_seconds

        self._safe_update(change)
        print(f"Proxy provider {provider.name} cooling down for {self.cooldown_seconds}s")

    def clear_cooldown(self, provider):
        def change(state):
            self._entry(state, provider).pop('cooldown_until', None)

        self._safe_update(change)

    def _safe_update(self, change):
        try:
            self._update(change)
        except OSError as e:
            print(f"WARNING: proxy chain state unavailable ({type(e).__name__})")
