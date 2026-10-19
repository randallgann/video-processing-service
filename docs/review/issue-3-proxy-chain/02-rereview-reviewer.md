# Issue 3 re-review — e6dea30

Not cleared: R10 still leaks short configured credentials.

proxy_chain.py:redact filters secrets with len(s) >= 4. Neither configuration nor URL parsing requires a minimum length. With synthetic http://abc:xyz@phone.example:8080, a DownloadError carrying bare username abc and password xyz is emitted unchanged in retry diagnostics and propagated unchanged to the caller. These are configured credentials, so R10 applies regardless of length. Remove the minimum-length exception (or provide equivalent complete protection). No real credentials were used.

Attached 02-reviewer-proxy-guards.py.txt exercises the real wrapper with fake YoutubeDL and isolated state. Independent run on e6dea30: 2 PASS, 1 FAIL. The original long-credential repro passes unchanged. The cap repro was adapted to expect ProxyCapacityError with wait=0 and to assert the held slot remains the sole slot; it passes. The added short-credential test fails on emitted logs/raised errors. See 02-reviewer-repro-output.txt.

Independent full shipped suite (excluding reviewer file): 171 passed in 0.29s. All five entry/modules compile. R9 code inspection confirms atomic cap checking even while ignoring cooldown, fail-closed on capped state errors, bounded monotonic wait, and preservation of full provider budget. Existing tests exercise release-during-wait and timeout. No network, real proxy, image build, deployment, or push performed. rag-widget 0c4383b unchanged from previous independent offline render.

Only remaining blocker: short-credential R10 regression. Job-time config validation remains lead-accepted. Please preserve reviewer artifacts and rerun gates on the fix head.
