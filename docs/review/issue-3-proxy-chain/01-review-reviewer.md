# Issue 3 independent review — 11e4b0b + 0c4383b

Not cleared. Two concrete findings, independently reproduced in detached worktrees. No implementation edited. Rework must also include the lead's R8 (cooled provider retains its attempts budget); review the new head.

## MUST-FIX: last resort exceeds concurrency cap

proxy_chain.py:201-207 puts both cooldown and max_concurrent checks inside `if not force`; ytdlp_opts.py:131-134 calls acquire(force=True) when every provider was skipped, including when full. Hold one live slot for a max_concurrent=1 provider and invoke the retry wrapper: the real state file records TWO slots before YoutubeDL construction. This defeats the configured cap under parallel jobs. The plan permits last resort for cooldown, not unlimited concurrency. Keep the cap in force even on last resort. Lead ruling requested for all-full handling (retryable capacity error or bounded wait).

Repro: 01-reviewer-proxy-guards.py.txt test_full_provider_cap_is_not_bypassed. It instruments the real slot state and fails at the second slot. A fix that rejects capacity before construction should add the expected retryable-error assertion to this fixture.

## MUST-FIX: retry diagnostics leak proxy credentials

ytdlp_opts.py:107-108 prints the first 80 characters of the raw DownloadError. Inject a retryable 429 exception containing a synthetic authenticated proxy URL: stdout contains both synthetic username and password. Provider/attempt-only logging is required; remove raw exception content from this diagnostic or sanitize it at the boundary. Check other emitted exception diagnostics for the same exposure without global logging suppression.

Repro: test_retry_diagnostic_never_emits_proxy_credentials. All credentials in this fixture and output are deliberately synthetic and never valid. Existing successful tests do not cover credential-bearing exception text.

## Independently verified

- Exact pinned transcriptor head 11e4b0b: full pytest 158 passed; cloud-vm-processor.py, captions.py, ytdlp_opts.py, proxy_chain.py and yt-dlp-aduio-processor-v1.py compile.
- Both reviewer tests FAIL on 11e4b0b (attached output); no external requests or real proxies used; isolated TMPDIR state.
- Exact pinned rag-widget head 0c4383b: diff against 9669699 contains the intended optional YTDLP_PROXY_CHAIN secret reference; kustomize build kubernetes/overlays/k3s exits 0. No deployment or real secrets read.
- Dockerfile.optimized includes proxy_chain.py; production call sites are unchanged.

A disclosed deviation remains for lead adjudication: invalid chain configuration is rejected at job-time rather than at startup as the plan says. This review does not authorize that plan change.

## Lead rulings and R8 addendum

Lead confirmed both MUST-FIX at 00:14Z: R9 caps never bypassed, bounded 60-second wait then retryable capacity error; R10 redact proxy credentials everywhere including DownloadError text. Preserve R8 full last-resort budget and captions total cap. Both attached reviewer tests independently fail again on d5b05b2 (output attached), so R8 alone does not close these defects. Adapt the capacity fixture to fake time and assert the bounded retryable failure with no excess slot or YoutubeDL construction; also cover a released slot becoming available within the wait.

Lead accepted job-time malformed-chain validation for P1 at 00:14Z; startup validation is a later item, not a blocker. Only R9 and R10 block this packet.
