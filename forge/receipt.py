"""Evidence receipt v2 (H04).

A completion receipt that identifies WHAT changed, WHAT checked it, and the EXACT
workspace state that was judged — so a change after verification deterministically
invalidates the receipt, identical workspaces produce identical manifests, and an
opaque (unattributed) mutation is never dressed up as a specific verified file list.

The workspace digest is git's content-addressed tree, computed over a TEMPORARY
index so the real index is never touched, when a git work tree is available; a
sorted content-hash manifest otherwise. Both are deterministic — identical content
yields an identical digest and any content change flips it. This is the
first-principles version of the stale-verification guard the H02 reducer approximates
with flags.
"""
import hashlib
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

RECEIPT_VERSION = 2

_SKIP_DIRS = frozenset((".git", "node_modules", "__pycache__", ".venv", "venv",
                        ".forge", ".mypy_cache", ".pytest_cache", ".tox", "dist", "build",
                        ".ruff_cache", ".idea", ".hg", ".svn"))
# Volatile tooling artifacts that flip with no real change (OS/editor/test tooling).
# Deliverables an agent may legitimately produce (build.log, .env, generated data)
# are NOT skipped — a change to those SHOULD invalidate a receipt.
_SKIP_FILE_NAMES = frozenset((".DS_Store", ".coverage"))
_SKIP_FILE_SUFFIXES = (".pyc", ".pyo", ".swp", ".swo", ".orig", "~")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_text(text: str) -> str:
    return _sha((text or "").encode("utf-8", "replace"))[:16]


def _skip_file(name: str) -> bool:
    return name in _SKIP_FILE_NAMES or name.endswith(_SKIP_FILE_SUFFIXES)


def workspace_digest(cwd: str) -> str:
    """A deterministic content digest of the workspace: every real file under `cwd`
    (excluding VCS internals, dependency/build directories, and volatile tooling
    artifacts) hashed and sorted.

    ONE backend regardless of git presence — so identical content ALWAYS yields an
    identical digest and any real change flips it, including a change to a gitignored
    DELIVERABLE, and a git subdirectory is scoped to its own subtree. (An earlier
    git-tree backend was dropped: it respected .gitignore, disagreed with the manifest,
    and could spuriously flip on a transient git hiccup.)"""
    cwd = os.path.realpath(cwd or ".")
    entries: List[str] = []
    for root, dirs, files in os.walk(cwd):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
        for fn in sorted(files):
            if _skip_file(fn):
                continue
            fp = os.path.join(root, fn)
            rel = os.path.relpath(fp, cwd)
            try:
                with open(fp, "rb") as f:
                    entries.append(rel + ":" + _sha(f.read()))
            except OSError:
                entries.append(rel + ":unreadable")
    return "ws1:" + _sha("\n".join(entries).encode())


@dataclass(frozen=True)
class Check:
    verifier: str          # what checked it, e.g. "run_tests" or the test command
    command: str
    exit_code: int
    output_digest: str
    timestamp: float
    version: str = ""      # verifier identity/version, when known

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceReceipt:
    claim: str
    contract_id: str = ""
    verified_workspace: Optional[str] = None      # the workspace digest a check judged
    changed_paths: List[Dict[str, str]] = field(default_factory=list)  # measured: [{path, digest}]
    opaque_changes: bool = False                  # an unattributed (bash) mutation happened
    checks: List[Check] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    unverified_assumptions: List[str] = field(default_factory=list)
    authority: Dict[str, Any] = field(default_factory=dict)
    approved: bool = False    # H05: a recorded approval accepted this as unverified
    version: int = RECEIPT_VERSION

    @property
    def checks_passed(self) -> bool:
        return bool(self.checks) and all(c.exit_code == 0 for c in self.checks)

    def verified(self, cwd: str) -> bool:
        """True only if a check PASSED and the workspace STILL matches the exact state
        that was checked — so any change (measured or opaque) after verification, in
        this or a later turn, deterministically invalidates the receipt."""
        if not self.checks_passed or not self.verified_workspace:
            return False
        return workspace_digest(cwd) == self.verified_workspace

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["checks"] = [c.to_dict() for c in self.checks]
        return d
