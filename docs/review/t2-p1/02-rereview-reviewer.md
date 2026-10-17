# T2-P1 focused re-review — clear

Reviewer: review-reviewer@llmexpert, 2026-10-17.
Target: youtube-video-transcriptor feat/t2-captions-worker @ 8f56f05,
delta from reviewed 5e0b600.

## Findings resolved

- Duration enforcement now distinguishes missing duration from missing metadata.
  With no usable probed duration, downloaded audio is measured and capped before
  splitting, chunk upload or Replicate. The caption path checks its duration
  before uploading the transcript. Over-cap work publishes failed and acks once;
  probe metadata is preserved on below-cap fallback paths.
- Rolling-caption dedupe keeps the displayed cue context separately from trimmed
  output. The overlap-then-growth chain now retains every new word once; added
  mixed-chain cases cover context resets, sliding windows, repeated openings,
  word boundaries, purity and idempotence.

## Independent verification

- `.venv/bin/python -m pytest -q`: exit 0, **67 passed**, Python 3.10.
  Includes F4 shape, pre-refactor Whisper regression, options preservation,
  quality dispatch, metadata, cap, ack, Docker COPY and SIGTERM tests.
- `.venv/bin/python -m py_compile cloud-vm-processor.py yt-dlp-aduio-processor-v1.py
  ytdlp_opts.py captions.py`: exit 0.
- Original reviewer `repro.py.txt` copied unchanged to the test directory:
  exit 0, **3 passed**. Observed audio-cap case: zero transcription calls, failed
  completion with 15000s/14400s limit text, one ack. Caption-cap case: failed,
  zero uploads. Rolling text: `a b c d e f g`, with no repeated window.
  Output saved in `repro-fixed-result.txt`; temporary test removed after running.
- The Pub/Sub >=2.13.0 floor remains adequate, independently checked against
  official v2.13.0 source in the first review. No dependency-floor change needed.

## Verdict and limits

**Clear for intake-lead closure at 8f56f05.** Both blocking findings are resolved;
no outstanding blocking findings in the reviewed T2-P1 scope. No implementation
was modified by the reviewer. This addendum and passing repro output are
uncommitted for the builder to preserve on this branch.

No real proxy/caption download, live Whisper fallback, image build, or real
Pub/Sub/pod SIGTERM drain was exercised. Those remain the lead's deploy checklist
items. No secrets read, deployment, remote push or PR performed; clearance grants
no authorization for those actions.
