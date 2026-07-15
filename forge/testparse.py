"""P6.3 — structured test-run digests.

Test output is the highest-value, worst-shaped observation in the loop: on a large run
the failure needle is buried in site-packages hay, and the observation cut frequently
slices off the part that matters — pytest's "short test summary info" lives at the END.
This module turns raw runner output into a compact digest the model can act on: a counts
line, then up to MAX_FAILS per-failure entries (test id + the assertion/error line) and a
few traceback frames pointing into the WORKSPACE (site-packages frames dropped).

Deterministic, stdlib-only. Serves the `run_tests` action, the opportunistic hook when the
model runs a known runner via bash, and (reusable) the fleet verifier's evidence field.
Returns None when nothing test-shaped is recognized, so callers keep the raw output.
"""
import os
import re

MAX_FAILS = 5

# runners recognized at a command HEAD — drives the opportunistic bash digest hook.
_RUNNER_RE = re.compile(
    r"^(pytest|py\.test|(?:python[0-9.]*|py) -m (?:pytest|unittest)|"
    r"go test|cargo test|npm (?:run )?test|yarn (?:run )?test|pnpm (?:run )?test|"
    r"jest|vitest|tox|rspec)\b")

# pytest short-summary lines (at the END): `FAILED path::test - reason`. The node id
# MUST contain `::` — so unittest's `FAILED (failures=1)` summary can't be mistaken for one.
_PYTEST_FAIL = re.compile(r"^(?:FAILED|ERROR)\s+(\S+::\S+)(?:\s+-\s+(.*))?$")
# pytest final banner: `===== 3 failed, 5 passed in 0.12s =====`
_PYTEST_SUMMARY = re.compile(r"^=+.*?\b\d+\s+(?:failed|passed|error|skipped)\b.*?=+\s*$")
# unittest per-failure header: `FAIL: test_x (module.Class)` / `ERROR: ...`
_UNITTEST_FAIL = re.compile(r"^(?:FAIL|ERROR):\s+(.+?)\s*$")
# an error / assertion message line (the reason), e.g. `AssertionError: -1 != 5`
_ERRLINE = re.compile(r"^(?:E\s+)?((?:\w+\.)*\w*(?:Error|Exception|Failure)\b.*|assert\b.*)$")
_UNITTEST_SUMMARY = re.compile(r"^(OK(?:\b.*)?|FAILED(?:\b.*)?|Ran \d+ tests?\b.*)$")
_UNITTEST_SEP = re.compile(r"^={20,}\s*$", re.M)
# go / cargo / jest failure lines
_GO_FAIL = re.compile(r"^\s*--- FAIL:\s+(\S+)")
_CARGO_FAIL = re.compile(r"^test\s+(\S+)\s+\.\.\.\s+FAILED\s*$")
_JEST_FAIL = re.compile(r"^\s*(?:✕|×|✗|●)\s+(.+?)\s*$")
# a python traceback frame
_FRAME = re.compile(r'File "([^"]+)", line (\d+)(?:, in (\S+))?')

# a run that collected ZERO tests, decided from the FINAL summary line only (searched
# from the END, like _summary): a nested runner leaking `Ran 0 tests` into a passing
# suite's output must not flip it, and a pytest collection ERROR ("collected 0 items /
# 1 error", terminal `N error in ...`) is positive evidence of breakage, NOT a zero run.
_RAN_N_RE = re.compile(r"^Ran (\d+) tests?\b")
_NO_TESTS_RAN_RE = re.compile(r"^=*\s*no tests ran\b", re.I)
_PYTEST_COUNTS_RE = re.compile(
    r"^(?:=+.*?)?\b\d+\s+(?:passed|failed|errors?|skipped|deselected|xfailed|xpassed|warnings?)\b")


def zero_collected(output):
    """True when the FINAL summary of runner output shows a run that collected/ran
    ZERO tests. Such a run verifies nothing, so it must never satisfy a verification
    gate — on Python < 3.12 unittest exits 0 on it, which reads as a passing suite
    to any exit-code check."""
    if not output:
        return False
    for ln in reversed(output.splitlines()):
        s = ln.strip()
        if not s:
            continue
        m = _RAN_N_RE.match(s)
        if m:
            return m.group(1) == "0"
        if _NO_TESTS_RAN_RE.match(s):
            return True
        if _PYTEST_COUNTS_RE.match(s):
            return False
    return False


def is_test_runner(command):
    """True if `command`'s head is a known test runner (opportunistic-digest hook)."""
    return bool(command and _RUNNER_RE.match(command.strip()))


def _summary(lines):
    """The most informative counts/summary line, searched from the END (where runners
    print it). pytest's `=== N failed ===` banner or unittest's OK/FAILED/Ran line."""
    for ln in reversed(lines):
        s = ln.strip()
        if _PYTEST_SUMMARY.match(s):
            return s.strip("= ").strip()
        if _UNITTEST_SUMMARY.match(s):
            return s
    return None


def _pytest_fails(lines):
    out = []
    for ln in lines:
        m = _PYTEST_FAIL.match(ln.strip())
        if m:
            out.append((m.group(1), (m.group(2) or "").strip()[:160]))
    return out


def _unittest_fails(text):
    out = []
    for block in _UNITTEST_SEP.split(text):
        blines = block.splitlines()
        hdr_i = next((i for i, l in enumerate(blines[:3]) if _UNITTEST_FAIL.match(l.strip())), None)
        if hdr_i is None:
            continue
        tid = _UNITTEST_FAIL.match(blines[hdr_i].strip()).group(1)
        # the reason is the last error/assertion line before the run's summary tail
        reason = ""
        for l in blines[hdr_i + 1:]:
            s = l.strip()
            if s.startswith(("Ran ", "OK", "FAILED")):   # reached the summary — stop
                break
            m = _ERRLINE.match(s)
            if m:
                reason = m.group(1)[:160]
        out.append((tid, reason))
    return out


def _other_fails(lines):
    out = []
    for ln in lines:
        for rx in (_GO_FAIL, _CARGO_FAIL, _JEST_FAIL):
            m = rx.match(ln)
            if m:
                out.append((m.group(1).strip(), ""))
                break
    return out


def _workspace_frames(lines, cwd):
    """Traceback frames pointing INTO the workspace (drop stdlib/site-packages noise)."""
    seen, out = set(), []
    for ln in lines:
        m = _FRAME.search(ln)
        if not m:
            continue
        path = m.group(1)
        try:
            rp = os.path.realpath(path)
        except (OSError, ValueError):
            rp = path
        if cwd and not rp.startswith(cwd):
            continue
        rel = os.path.relpath(rp, cwd) if cwd else path
        key = (rel, m.group(2))
        if key in seen:
            continue
        seen.add(key)
        fn = f" in {m.group(3)}" if m.group(3) else ""
        out.append(f"{rel}:{m.group(2)}{fn}")
    return out


def digest(output, cwd=""):
    """Compact structured digest of raw test output, or None if nothing test-shaped is
    recognized (caller keeps the raw). Counts line + up to MAX_FAILS failures + a few
    workspace traceback frames."""
    if not output or not output.strip():
        return None
    lines = output.splitlines()
    real_cwd = ""
    if cwd:
        try:
            real_cwd = os.path.realpath(cwd)
        except (OSError, ValueError):
            real_cwd = ""

    summary = _summary(lines)
    fails = _pytest_fails(lines) or _unittest_fails(output) or _other_fails(lines)
    if not summary and not fails:
        return None                      # not test-shaped — let the caller keep raw

    parts = [summary or "test run (no summary line found)"]
    if fails:
        extra = f" (showing {MAX_FAILS} of {len(fails)})" if len(fails) > MAX_FAILS else ""
        parts.append(f"\n{len(fails)} failing{extra}:")
        for tid, reason in fails[:MAX_FAILS]:
            parts.append(f"  ✗ {tid}" + (f" — {reason}" if reason else ""))
    frames = _workspace_frames(lines, real_cwd)
    if frames:
        parts.append("\nin your files:")
        parts.extend("  " + f for f in frames[:MAX_FAILS])
    return "\n".join(parts)
