# Captions fail-fast hotfix cleared — fade929

No blocking findings against HOTFIX_CAPTIONS_FAILFAST.md at c268fc5. Reviewed only the owner-approved transcriptor exception, branch fix/captions-fail-fast off main8b13298.

## Independent verification

Pinned detached worktree /tmp/captions-hotfix-review-fade929:
- Python3.10.20 via the repo venv: python -m pytest -q ->128 passed, zero failures.
- python -m py_compile cloud-vm-processor.py captions.py ytdlp_opts.py ->exit0.
- Four independent combined-path tests invoke real fetch_captions and the real retry wrapper with only YoutubeDL/options/sleep replaced. Both the metadata-probe path (info absent) and track-download path (info supplied) make exactly one attempt by default, exactly three with CAPTIONS_ATTEMPTS=3, and return None on exhausted429 for Whisper fallback. The fixture initially patched the wrong module for locally imported option helpers; corrected to ytdlp_opts and reran4/4. No implementation failure.
- Source inspection: call-time env helpers, false/0/no/off case-insensitive kill switch, floor1/garbage handling, both captions extraction call sites pass the override. Processor disabled path skips captions and retains duration probe/Whisper completion; full-suite regression tests verify enabled/disabled/Whisper variants and one disabled log.
- Diff confirms the downloader, proxy pool logic and proxy_for_attempt unchanged; metadata probe still omits the override. Dockerfile.optimized unchanged and still copies captions.py/ytdlp_opts.py; no new production module. Existing two fake_extract signature changes accommodate the new keyword without weakening their assertions.

## Limits / retained finding

No live YouTube/proxy/PubSub/Replicate request, credentials, image build, push or deployment. Attempt budget is not a wall-clock timeout; live latency remains the lead's deployment check. The metadata probe still retains the original ten-attempt budget and can delay either quality; this is explicitly parked outside this approved hotfix, not a blocker for the specified change. Tests disable dotenv loading and use synthetic inputs.

Clear to intake-lead for closure. Builder may commit these docs-only review artifacts on the branch.
