"""Full-fidelity workspace fixtures (H11).

Replay already reproduces the model-output and harness-decision half of a recorded
run. This captures the ENVIRONMENT half: a content-addressed snapshot of the workspace
files, restorable into a THROWAWAY directory so a mutation/test trace can be re-driven
against the same inputs — never against the original workspace.

Two hard rules from the roadmap non-goals:
  * NEVER archive credentials. Secret files (.env*, private keys, .ssh/.aws/.gnupg, …)
    are EXCLUDED, recorded only as a named fidelity limitation — never their contents.
    Matching is case-insensitive (Deploy.PEM leaks the same key .pem does), and
    symlinks are never followed — a link named innocently can point at ~/.aws.
  * Bound the capture. Files over a size cap, junk/dependency directories, and
    anything past an AGGREGATE budget are excluded (also as fidelity limitations),
    so a fixture can't balloon, OOM the process, or drag in a node_modules tree.

A missing or excluded input is reported explicitly (`fidelity_limitations`) rather than
silently faked, so a replay's fidelity is always honest.
"""
import base64
import hashlib
import json
import os
import stat
from typing import Any, Dict, List

FIXTURE_VERSION = 1
MAX_FILE_BYTES = 256 * 1024          # per-file cap; larger files are excluded, not archived
MAX_TOTAL_BYTES = 32 * 1024 * 1024   # aggregate content budget — bounds memory AND fixture size
MAX_FILES = 4000                     # aggregate file-count budget

# reuse H04's directory skip list so capture and the workspace digest agree
from .receipt import _SKIP_DIRS, _skip_file

# All secret matching is done on LOWERCASED names — macOS's default filesystem is
# case-insensitive, so Deploy.PEM and .ENV are the same credentials .pem/.env are.
_SECRET_DIRS = frozenset((".ssh", ".aws", ".gnupg", ".gpg"))
_SECRET_NAMES = frozenset((".env", ".netrc", ".npmrc", ".pgpass", ".git-credentials",
                           "credentials", "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"))
# ".env" as a SUFFIX too: prod.env / secrets.env (docker-compose env_file convention).
_SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".keystore", ".jks", ".env")
_SAFE_ENV = frozenset((".env.example", ".env.sample", ".env.template", ".env.dist"))


def _is_secret(rel: str) -> bool:
    """A file whose CONTENTS must never be archived (a credential). Case-insensitive."""
    parts = rel.replace("\\", "/").lower().split("/")
    if any(p in _SECRET_DIRS for p in parts):
        return True
    base = parts[-1]
    if base in _SAFE_ENV:
        return False                                   # templates are not secrets
    if base in _SECRET_NAMES or base.startswith(".env"):
        return True
    return base.endswith(_SECRET_SUFFIXES)


def _digest(files: Dict[str, Dict[str, str]]) -> str:
    """Content digest with unambiguous framing: JSON-encode the sorted entries so no
    crafted file content can collide with a different file SET (a ':'/newline-joined
    concatenation could — 'a: x\\nb:utf-8:y' vs files a+b)."""
    payload = json.dumps(
        [[rel, e.get("encoding", ""), e.get("content", ""), bool(e.get("exec"))]
         for rel, e in sorted(files.items())],
        separators=(",", ":"))
    return "wsf1:" + hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()


def capture(cwd: str, max_file_bytes: int = MAX_FILE_BYTES,
            max_total_bytes: int = MAX_TOTAL_BYTES, max_files: int = MAX_FILES) -> Dict[str, Any]:
    """A content-addressed fixture of the workspace under `cwd`. Secret files, symlinks,
    non-regular files (FIFOs/sockets), files over the size cap, and anything past the
    aggregate budget are excluded — recorded as fidelity limitations, never archived."""
    cwd = os.path.realpath(cwd or ".")
    files: Dict[str, Dict[str, Any]] = {}
    excluded: List[Dict[str, Any]] = []
    total = 0
    over_budget = 0
    for root, dirs, names in os.walk(cwd):
        keep = []
        for d in sorted(dirs):
            dp = os.path.join(root, d)
            rel = os.path.relpath(dp, cwd)
            if d.lower() in _SECRET_DIRS:              # record pruned secret dirs (.ssh/.aws/…)
                excluded.append({"path": rel, "reason": "secret-dir"})
            elif os.path.islink(dp):                   # never follow; never silently omit
                excluded.append({"path": rel, "reason": "symlink-dir"})
            elif d in _SKIP_DIRS:
                pass                                   # junk/dependency dir — not fidelity-relevant
            else:
                keep.append(d)
        dirs[:] = keep
        for fn in sorted(names):
            fp = os.path.join(root, fn)
            rel = os.path.relpath(fp, cwd)
            if _skip_file(fn):
                continue                               # volatile junk — not fidelity-relevant
            if _is_secret(rel):
                excluded.append({"path": rel, "reason": "secret"})
                continue
            try:
                st = os.lstat(fp)                      # lstat: NEVER dereference a link
            except OSError:
                excluded.append({"path": rel, "reason": "unreadable"})
                continue
            if stat.S_ISLNK(st.st_mode):
                # a link named innocently can point at ~/.aws/credentials — never read through it
                excluded.append({"path": rel, "reason": "symlink"})
                continue
            if not stat.S_ISREG(st.st_mode):
                # a FIFO/socket would block open()/read() forever
                excluded.append({"path": rel, "reason": "not-a-regular-file"})
                continue
            if st.st_size > max_file_bytes:
                excluded.append({"path": rel, "reason": "too-large", "bytes": st.st_size})
                continue
            if len(files) >= max_files or total + st.st_size > max_total_bytes:
                over_budget += 1                       # aggregate budget — bound memory + size
                continue
            try:
                with open(fp, "rb") as f:
                    data = f.read()
            except OSError:
                excluded.append({"path": rel, "reason": "unreadable"})
                continue
            total += len(data)
            try:
                entry: Dict[str, Any] = {"encoding": "utf-8", "content": data.decode("utf-8")}
            except UnicodeDecodeError:
                entry = {"encoding": "base64", "content": base64.b64encode(data).decode("ascii")}
            if st.st_mode & stat.S_IXUSR:
                entry["exec"] = True                   # a lost +x diverges ./script.sh replays
            files[rel] = entry
    if over_budget:
        excluded.append({"path": f"({over_budget} more file(s))", "reason": "capture-budget"})
    fixture = {"version": FIXTURE_VERSION, "files": files, "excluded": excluded}
    fixture["digest"] = _digest(files)
    return fixture


def restore(fixture: Dict[str, Any], dest: str) -> str:
    """Write a fixture's files into `dest` (created if absent). NEVER pass the original
    workspace as dest — a fixture restores into a throwaway directory."""
    dest = os.path.realpath(dest)
    os.makedirs(dest, exist_ok=True)
    for rel, entry in (fixture.get("files") or {}).items():
        fp = os.path.realpath(os.path.join(dest, rel))
        if not fp.startswith(dest + os.sep) and fp != dest:
            continue                                    # refuse a path escaping dest (defensive)
        os.makedirs(os.path.dirname(fp) or dest, exist_ok=True)
        data = (entry["content"].encode("utf-8") if entry.get("encoding") == "utf-8"
                else base64.b64decode(entry.get("content", "")))
        with open(fp, "wb") as f:
            f.write(data)
        if entry.get("exec"):
            os.chmod(fp, 0o755)                         # restore the executable bit
    return dest


def fidelity_limitations(fixture: Dict[str, Any]) -> List[str]:
    """Human lines naming what the fixture could NOT capture — so a replay against it is
    honestly labeled rather than silently lower-fidelity."""
    lines = []
    for e in fixture.get("excluded") or []:
        extra = f" ({e['bytes']} bytes)" if e.get("bytes") else ""
        lines.append(f"{e['path']}: {e['reason']}{extra}")
    return lines
