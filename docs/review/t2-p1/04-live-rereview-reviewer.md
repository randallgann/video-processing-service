# T2-P1 permanent-failure live rework — locally cleared

Reviewed b050815 against 221a2c0, youtube-video-transcriptor feat/t2-captions-worker. Origin: lead packet qitem-20261017173114-b2273fafe3f7e91d, which explicitly requires age/private/removed/members-only/geo/unavailable errors to fail and ACK. No blocking findings in this focused rework.

The classifier folds case and apostrophes, distinguishes permanent errors from bot checks, and preserves transient throttling overrides. A permanent probe error now raises NonRetryableError immediately; a permanent download failure does likewise. The existing terminal path publishes failed progress/completion before acknowledging, preserving the original error and correlation. Transient errors retain the existing redelivery behavior, while the existing exhausted bot-check handling is preserved. No retry-loop, captions, Whisper or billing implementation was changed by this commit.

## Independent checks

- .venv/bin/python -m pytest -q: 98 passed in 0.07 seconds, including the full ten-proxy 429 retry test, age-gated ACK, permanent probe short-circuit, transient probe fallback, duration cap, rolling-caption, selected-track and Whisper compatibility cases.
- .venv/bin/python -m py_compile cloud-vm-processor.py yt-dlp-aduio-processor-v1.py captions.py ytdlp_opts.py: exit 0.
- Eight additional reviewer process_message cases: 8 passed. Age/private/removed/members-only/geo download errors publish exactly one failed completion with their error and ACK; both throttling wordings and an explicit 429 publish failed but leave the message unacknowledged for redelivery. All avoid transcription. Fixture: review-failure-paths.py.txt (imports the existing worker fixture); output: 04-reviewer-failure-paths-output.txt. Temporary executable fixture removed afterward.
- HEAD remained b050815 during verification. No implementation edits by review.

Geo-blocking is permanent because the lead explicitly selected that policy, despite possible exit-country variation. Bare video unavailable is also permanent as requested; the two identified throttling variants override it. Unknown error text remains retryable.

Live draining of the previously redelivered age-gated message has not been verified. Lead/owner must redeploy and observe failed completion plus ACK/no subsequent redelivery. The builder reports the lead's newer live capture showed all ten proxy attempts on 221a2c0; that is reported live evidence, not independently observed by this review. No deployment, real Pub/Sub, proxy network access or secrets access was performed. Review artifacts are uncommitted for builder to preserve.
