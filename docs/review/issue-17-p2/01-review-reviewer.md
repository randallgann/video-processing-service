# Issue 17 P2 independent review — 8056f4a

Cleared for lead closure/live verification at youtube-video-transcriptor feat/issue17-nack-final 8056f4a on c0c762e. Complete two-file diff reviewed against ISSUE_17_19_FAILURE_SEMANTICS.md D1/D2 and P2. No blocking findings; no implementation edited.

Retryable pre-transcription failure with remaining attempts publishes failed/final:false progress at retrying, emits no completion result and explicitly nacks. Nonretryable errors, exhausted budget and failures after transcription results are final: final:true progress, failed completion, ack. Existing success/invalid-message ack sites now clear the fallback counter. The JSON max_attempts key is correct despite internal kwarg attempt_limit (needed to avoid an existing textual captions guard). API P1 treats absent final as true, so deploy api-service first as planned.

The positive integer budget defaults to5, invalid settings warn/fallback. Pub/Sub delivery_attempt takes precedence; otherwise failed deliveries are counted by stable message_id with an oldest-entry bound1000 and cleanup on ack. No proxy/captions/bitrate code changes. Final-decision logging includes attempt, limit, final and ack/nack action.

Fresh independent checks at exact detached head8056f4a in the lead-granted slot, sequentially:
- Repo .venv Python3.10 -m py_compile cloud-vm-processor.py: exit0.
- Same Python -m pytest -q: exit0,200 passed/0failed/0skipped (0.36s).
- Reviewed meaningful regressions: fifth failure finalizes after four nacks; SDK attempt2/5 overrides fallback; per-id counting, env limit, bounded state and ack cleanup; final/nonretryable/after-transcription behavior; progress and completion JSON fields; success after retry clears state.
- Inspected the actual installed Pub/Sub Message.nack source: it enqueues NackRequest; Message.delivery_attempt property exists. This is local library evidence, not a live subscription test. Import emits existing Python3.10 support warnings, no test failure.

Known plan limitations: fallback state is process-local and oldest entries may be evicted; restarts reset the budget. Delivery can land on another subscriber. D4 dead-letter policy supplies broker attempts and retry policy supplies backoff; without backoff short outages may consume5 attempts rapidly. These are not hidden guarantees of durable global counting. Lead's D4 decision remains important.

No live lease/FlowControl, Pub/Sub redelivery/backoff, final-refund ledger or worker rollout verified. Full tests use fake external collaborators. The consumer-side P1 has separately cleared at964f7a0 with delayed/in-flight guards; deploy that api-service before this worker. No image build, deploy, push, real content mutation or secret access. Slot released; docs untracked for builder to read and commit.
