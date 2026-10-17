# T2-P1 review — rework requested

Reviewer: review-reviewer@llmexpert, 2026-10-17.
Target: youtube-video-transcriptor feat/t2-captions-worker @ 5e0b600
against main 53380e8. Scope: T2_BASELINE.md section 6, F3/F4, R4/section 9.

## Blocking findings

1. **MUST-FIX — missing duration in a successful probe bypasses the cap.**
   `cloud-vm-processor.py:1388` runs its audio-duration backstop only when the
   entire info object is false. A valid metadata object with no duration is true,
   so the worker downloads and transcribes an over-cap video. The measured audio
   duration is only obtained later for completion metadata, after Replicate spend.
   Repro: truthy probe info without duration; audio duration 15000; cap 14400;
   quality whisper -> one transcription, completed with duration_seconds=15000,
   message acked. The declared worker cap is not enforced.

   The captions path has the same hole: probe lacks duration, fetch_captions
   reports duration_seconds=15000, but the worker uploads and reports completed
   without checking that known caption duration against the cap.

   Key the fallback on the absence of a usable duration, not absence of all
   metadata. Enforce measured audio duration before chunk upload/Replicate spend
   when the probe cannot supply it; enforce caption metadata duration before
   uploading/accepting captions. Over-cap jobs must publish failed and ack once.
   Retain below-cap/known-duration behavior and metadata. Lead explicitly confirms
   this is a money bug and MUST-FIX.

2. **HIGH — chained rolling captions repeat text after a trimmed overlap.**
   `captions.py:169-170` compares each new cue with the previous OUTPUT cue,
   whose overlapping prefix may already have been trimmed at lines 180-187.
   That loses the rolling window needed to recognize the next growing cue.
   Input: `a b c d` -> `c d e f` -> `c d e f g`.
   Actual joined output: `a b c d e f c d e f g`.
   Expected: `a b c d e f g` (preserve every new word once).
   This is the sliding-window-then-prefix-growth combination missing from the
   two-cue table; the current function can be idempotent while leaving repeated
   speech in the actual transcript.

   Preserve sufficient original rolling context or assembled suffix context so
   trimming one output cue does not defeat the next comparison. Add 3+ cue mixed
   overlap/growth regressions; retain distinct speech, word boundaries, time spans,
   purity, and idempotence. Do not fix by dropping the newly introduced words.

## Runnable evidence

Copy `docs/review/t2-p1/repro.py.txt` to `tests/test_review_regressions.py`, then:

```
.venv/bin/python -m pytest -q -s tests/test_review_regressions.py
```

At 5e0b600: exit 1, **3 failed** (audio cap, rolling chain, caption cap).
Full output is `docs/review/t2-p1/repro-result.txt`.
The tests drive the real process_message with the existing offline worker fixture
and real dedupe_rolling; no service/network calls. Temporary test file removed
after verification. Review artifacts remain uncommitted; implementation untouched.

## Independent checks and decisions

- `.venv/bin/python -m pytest -q`: exit 0, **55 passed** under Python 3.10 with
  installed dependencies. Added repro tests fail separately as recorded above.
- `.venv/bin/python -m py_compile cloud-vm-processor.py yt-dlp-aduio-processor-v1.py
  ytdlp_opts.py captions.py`: exit 0. Both new modules are copied by Dockerfile.optimized.
- F4 shape and Whisper output regression pass; shared window loop preserves the
  existing format. Shared audio options retain the baseline dict and proxy/cookie
  behavior. Metadata, quality dispatch, fallback, correlation echo, bot-check ack,
  logged throttle counter and Docker COPY changes were inspected with their tests.
- **Pub/Sub floor adequate.** I checked the official [v2.13.0 subscriber source](https://github.com/googleapis/python-pubsub/blob/v2.13.0/google/cloud/pubsub_v1/subscriber/client.py):
  subscribe already accepts await_callbacks_on_shutdown and documents that true
  waits for running callbacks. Thus the required option existed at or before the
  declared >=2.13.0 floor; no dependency-floor change is required. The installed
  2.43.0 shutdown source stops consumption, awaits callbacks, then stops the leaser
  and dispatcher. Keeping main inside its original result() with the new option
  is consistent with the intended drain; no second result call is needed.
- Regex language variants, skipping translated auto tracks, raw probe metadata
  with formatted upload date, synthetic fixtures and no-op dotenv in tests are
  reasonable within this scope. The overlap algorithm's deviation is acceptable
  in principle but must satisfy finding 2. The probe fallback must satisfy finding 1.

## Not verified

No real proxy/caption download, live Whisper fallback, image build, real Pub/Sub
shutdown/ack drain or pod SIGTERM was performed. Those remain deploy-checklist
items. No secrets were read; no deploy, push or PR performed. No rag-widget
implementation changes. Return both fixes for focused re-review.
