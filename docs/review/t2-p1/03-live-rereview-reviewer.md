# T2-P1 live-check rework — locally cleared

Reviewed youtube-video-transcriptor feat/t2-captions-worker at 415a04b against deployed predecessor 4342ebf. No blocking code findings in this focused rework. Forward to intake-lead to repeat the live caption check.

The worker passes its existing probe metadata to fetch_captions. Track selection now precedes caption downloading; a usable track produces an escaped exact-code subtitleslangs option, and no usable track returns without a caption download. Standalone callers first obtain metadata without subtitle write flags. Shared probe options preserve network settings and the existing audio option regression remains green. Errors still fall back to Whisper.

## Independent verification

- .venv/bin/python -m pytest -q: 77 passed in 0.07 seconds (a second verification run also passed). Includes exact-code options, probe reuse, no-track behavior, Whisper compatibility, duration cap and rolling-caption regressions.
- .venv/bin/python -m py_compile cloud-vm-processor.py yt-dlp-aduio-processor-v1.py captions.py ytdlp_opts.py: exit 0. An initial compile command named a nonexistent optimized entry file; corrected to the two actual entry files above and verified all four successfully.
- Installed yt-dlp's real YoutubeDL.process_subtitles was called offline with automatic tracks en, en-en, en-de, en-US and en-orig and the new options. The selected map contained en only. This verifies the actual library selector rather than merely the test's regex approximation; no extraction/download/network occurred. The installed implementation prioritizes normal tracks and expands requested languages using orderedSet_from_options.
- Independently loaded 4342ebf:ytdlp_opts.py in a temporary directory and ran the new full-budget 429 regression against that old module. It passed: ten distinct pinned proxies, success on attempt ten, and exhaustion after ten when all attempts fail. yt-dlp was replaced with the test fake and sleeps disabled. Thus the explicit subtitle HTTP Error 429 path already had the full retry budget before this change; the new broader Too Many Requests marker adds coverage for text without the status code.
- Dockerfile.optimized already copies captions.py and ytdlp_opts.py; no new module or dependency is introduced.

The exact-code selection limits caption tracks per extraction attempt. Retries and yt-dlp's internal requests mean this is not a claim of one HTTP request total per video under failure.

No implementation edits, image build, deployment, live proxy use or secrets access. The observed live three-attempt stop is not explained by the reproduced explicit-429 path; later log lines/error text and runtime retry configuration are needed to identify that failure. Do not claim that live caption success or the proxy-pool 429 issue is fixed from this offline proof. Owner/lead repeats the live check. The review addendum remains uncommitted for builder to preserve.
