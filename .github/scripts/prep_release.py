#!/usr/bin/env python3
"""Prepare a release: compute the next version, bump it, keep the version-tied docs
in sync, and write the `release:` commit message from the commits since the last tag.

Run by .github/workflows/release.yml on a one-click dispatch. All logic here (pure
functions below) so it's unit-tested in tests/test_prep_release.py rather than trapped
in YAML. stdlib only — no deps, matching forge.

Usage: prep_release.py <bump: auto|patch|minor|major> [summary]
  Reads forge/__init__.py + `git log <last-tag>..HEAD`, writes:
    - forge/__init__.py         (new version)
    - SECURITY.md               (supported-versions table → new minor line)
    - .github/RELEASE_NOTES.md  (the `release:` commit message; used by the workflow)
  and prints the new version to stdout.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INIT = os.path.join(ROOT, "forge", "__init__.py")
SECURITY = os.path.join(ROOT, "SECURITY.md")
NOTES = os.path.join(ROOT, ".github", "RELEASE_NOTES.md")

# Which section a commit falls under in the notes (order = display order).
_SECTIONS = [("feat", "New"), ("fix", "Fixed"), ("docs", "Docs"),
             ("refine", "Refined"), ("perf", "Performance")]


# ---- pure logic (unit-tested) ----------------------------------------------

def parse_version(text):
    m = re.search(r'__version__\s*=\s*["\'](\d+)\.(\d+)\.(\d+)["\']', text)
    if not m:
        raise ValueError("no __version__ = \"X.Y.Z\" found")
    return tuple(int(g) for g in m.groups())


def bump_version(current, kind):
    """(major, minor, patch) tuple → next 'X.Y.Z' string for kind in major/minor/patch."""
    major, minor, patch = current
    if kind == "major":
        return f"{major + 1}.0.0"
    if kind == "minor":
        return f"{major}.{minor + 1}.0"
    if kind == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"unknown bump kind: {kind}")


def detect_bump(subjects):
    """Semver bump implied by conventional-commit subjects since the last tag:
    a breaking change → major, any feat → minor, otherwise patch (never 'none' —
    a release was explicitly requested, so it always advances)."""
    if any("!:" in s or "BREAKING CHANGE" in s or s.lower().startswith("breaking")
           for s in subjects):
        return "major"
    if any(re.match(r"feat(\(|:|!)", s) for s in subjects):
        return "minor"
    return "patch"


def set_version(init_text, new_version):
    return re.sub(r'(__version__\s*=\s*["\'])\d+\.\d+\.\d+(["\'])',
                  rf'\g<1>{new_version}\g<2>', init_text, count=1)


def update_security(security_text, new_version):
    """Point the supported-versions table at the new minor line (X.Y.x supported,
    < X.Y unsupported). Idempotent; leaves the rest of the doc untouched."""
    major, minor = new_version.split(".")[:2]
    text = re.sub(r'\|\s*\d+\.\d+\.x\s*\|\s*:white_check_mark:\s*\|',
                  f'| {major}.{minor}.x   | :white_check_mark: |', security_text, count=1)
    text = re.sub(r'\|\s*<\s*\d+\.\d+\s*\|\s*:x:\s*\|',
                  f'| < {major}.{minor}   | :x:                |', text, count=1)
    return text


def group_changes(subjects):
    """Bucket conventional-commit subjects into ordered (heading, [lines]) sections."""
    out = []
    for prefix, heading in _SECTIONS:
        lines = []
        for s in subjects:
            m = re.match(rf"{prefix}(?:\([^)]*\))?!?:\s*(.+)", s)
            if m:
                lines.append(m.group(1).strip())
        if lines:
            out.append((heading, lines))
    return out


def render_notes(new_version, subjects, summary=""):
    """The full `release:` commit message: a summary subject line + grouped body."""
    head = f"release: v{new_version}"
    if summary:
        head += f" — {summary}"
    body = []
    for heading, lines in group_changes(subjects):
        body.append(f"{heading}:")
        body += [f"  · {ln}" for ln in lines]
        body.append("")
    if not body:
        body = [f"Release v{new_version}.", ""]
    return head + "\n\n" + "\n".join(body).rstrip() + "\n"


# ---- git glue (thin; not unit-tested) --------------------------------------

def _last_tag():
    try:
        return subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0", "--match", "v*"],
            cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        return None


def _subjects_since(tag):
    rng = f"{tag}..HEAD" if tag else "HEAD"
    out = subprocess.check_output(
        ["git", "log", rng, "--no-merges", "--format=%s"], cwd=ROOT, text=True)
    return [ln for ln in out.splitlines() if ln.strip()]


def main(argv):
    bump = (argv[0] if argv else "auto").strip()
    summary = argv[1].strip() if len(argv) > 1 else ""

    with open(INIT) as f:
        init_text = f.read()
    current = parse_version(init_text)
    subjects = _subjects_since(_last_tag())

    if bump == "auto":
        bump = detect_bump(subjects)
    new_version = bump_version(current, bump)

    with open(INIT, "w") as f:
        f.write(set_version(init_text, new_version))
    if os.path.exists(SECURITY):
        with open(SECURITY) as f:
            sec = f.read()
        with open(SECURITY, "w") as f:
            f.write(update_security(sec, new_version))
    with open(NOTES, "w") as f:
        f.write(render_notes(new_version, subjects, summary))

    sys.stdout.write(new_version)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
