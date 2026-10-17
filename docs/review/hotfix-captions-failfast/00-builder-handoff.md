# Hotfix captions fail-fast — builder handoff (`fix/captions-fail-fast`)

Branch off `main` @ 8b13298. Plan: llmexpert-platform `docs/milestones/HOTFIX_CAPTIONS_FAILFAST.md` @ c268fc5. This is an owner-approved scope exception for this packet only. Nothing was pushed, built or deployed, and no live YouTube or proxy call was made.

## Results
- `.venv/bin/python -m pytest -q` (Python 3.10.20): **128 passed in 0.09s**. The baseline was 98, so 30 tests are new; 0 failed.
- `python3 -m py_compile cloud-vm-processor.py captions.py ytdlp_opts.py`: clean. The three modules also parse with `feature_version=(3,10)` on `python:3.10-slim`.
- `Dockerfile.optimized`: unchanged, with no new module. The only new tracked files are under `docs/`.
- **Mutation check: 8/8 caught,** each restored and checked with `cmp`:
  - the override ignored, or not floored at 1;
  - the probe call, or the download call, missing the captions budget;
  - a default of 10 instead of 1;
  - the kill switch ignored;
  - "disabled" falling into the "probe failed" log;
  - a case-sensitive switch.

## What shipped
1. **`ytdlp_opts.extract_with_bot_check_retry(..., attempts=None)`.** When `attempts` is given it replaces `YTDLP_BOT_CHECK_ATTEMPTS`, floored at 1, and an unproxied call still makes 1 attempt. Callers that don't pass it behave exactly as before.
2. **`captions.captions_attempts()`** reads `CAPTIONS_ATTEMPTS`: default 1, floor 1, and garbage gives 1. **`captions.captions_enabled()`** reads `CAPTIONS_ENABLED`: default true; `false`/`0`/`no`/`off` in any case (whitespace trimmed) turn it off. Both read the env at call time.
3. **`fetch_captions`** passes `attempts=captions_attempts()` on **both** extraction calls: the probe when `info is None`, and the track download. A proxied 429 or bot check now makes one attempt, then returns `None` and falls back to Whisper.
4. **Processor:**
   - The standard branch is now `if quality == 'standard' and info and captions_enabled():`.
   - When disabled, a new `elif quality == 'standard' and info:` logs **"Captions disabled (CAPTIONS_ENABLED=false); using Whisper"** once and continues to the download, with `transcript_source="whisper"`.
   - The existing probe-failed branch is unchanged. Without the new branch, "disabled" would have logged "Metadata probe failed".
   - The probe still runs, and `whisper` quality is unchanged.
5. **`CLAUDE.md`:** both variables are in "Optional (with defaults)", plus the contract sentence. No overlay change, since the defaults give the new behaviour.

**Untouched:** proxy pool logic, `proxy_for_attempt`, the audio download's budget (`yt-dlp-aduio-processor-v1.py`), the processor's metadata probe budget, rag-widget and chat-copilot. A test asserts that neither the downloader nor the processor passes `attempts=`.

## Tests (new)
- **`test_ytdlp_opts.py`:**
  - `attempts=1` on a proxied 429 makes exactly 1 attempt, with the env budget at 10;
  - without the keyword, the env budget of 3 makes 3 attempts;
  - override 2 → 2 attempts and override 0 → 1; unproxied with override 5 → 1.
- **`test_captions.py`:**
  - the `captions_attempts` table (7 cases) and the `captions_enabled` table (10 cases);
  - both `fetch_captions` calls receive the budget (1 by default, 2 from the env), with and without `info`;
  - the call-site guard for the downloader and the processor.
- **`test_processor.py`:**
  - disabled (`false`/`OFF`/`0`/`no`): no `fetch_captions`, the probe runs, download and transcribe run, completion is `whisper`, the log line appears exactly once, and there is no "probe failed" line;
  - `CAPTIONS_ENABLED=true` still uses captions;
  - `whisper` quality is unaffected and logs nothing about captions.
- **Existing tests (one edit):** two `fake_extract` stubs in `test_captions.py` gained `attempts=None` in their signature. Without it, the new keyword raises `TypeError` inside `fetch_captions`, which returns `None`. The assertions are unchanged, and the other 96 tests are untouched.

## Not verified
- **No live run:** no YouTube, proxy, Pub/Sub or Replicate call, no image build, and nothing on the cluster. The ~30 s captions-then-Whisper timing on the pod is the lead's live check.
- **Plan finding, not in this packet:** the processor's metadata probe still walks the 10-attempt pool for both qualities. If the live check still shows minutes at "download 0%", that is the next budget to bound.
