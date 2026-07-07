"""File-state ledger — the harness-owned model of what the agent has read and
still holds in context (P4.1).

Replaces the bare `read_files` set. Keyed by realpath, each entry tracks the
file's stat signature at read time (mtime, size, sha1), which step read it, the
line spans actually seen, whether its content is still in the model's context
(in_context), and a bounded RAM cache of the content for diff-since-last-read.

The ledger is the SINGLE per-path metadata store: the read-before-edit gate, the
served-from-cache read interception, and compaction eviction all consult it — so
a later structural-compaction pass (P4.2) can share it rather than build a second.

All I/O is stdlib (os.stat, open, hashlib, difflib). No third-party imports.
"""
import difflib
import hashlib
import os

CONTENT_CAP = 200_000      # max cached bytes for a single file's content
TOTAL_CAP = 2_000_000      # max cached bytes across ALL files (LRU-evict content, keep metadata)
DEFAULT_READ_LIMIT = 800   # mirrors tools.read_file's default `limit` so spans match what was shown
DIFF_CAP = 120             # max lines of a served unified diff


def _sha1(text):
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()


def _nlines(text):
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def _merge(spans):
    """Union a list of (start, end) inclusive line ranges."""
    if not spans:
        return []
    out = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


class Entry:
    __slots__ = ("realpath", "mtime", "size", "sha1", "read_step", "spans",
                 "whole", "in_context", "content", "obs_msg", "_seq", "_clen")

    def __init__(self, realpath):
        self.realpath = realpath
        self.mtime = None          # os.stat mtime at record time (None = poisoned/unknown → treated as changed)
        self.size = None
        self.sha1 = None
        self.read_step = None
        self.spans = []            # list of (start, end) 1-based inclusive line ranges the model has SEEN
        self.whole = False         # True if a whole-file read/write covered the file
        self.in_context = False    # is this file's content still in the message log?
        self.content = None        # cached text for diffing (None = never cached or LRU-dropped)
        self.obs_msg = None        # identity handle on the message dict holding this read's observation
        self._seq = 0              # LRU tick (higher = more recently touched)
        self._clen = 0             # bytes currently counted against TOTAL_CAP for this entry


class Ledger:
    def __init__(self):
        self.entries = {}
        self._tick = 0
        self._cached_bytes = 0

    # ---- lookups ----
    def get(self, realpath):
        return self.entries.get(realpath)

    def _stat(self, realpath):
        try:
            st = os.stat(realpath)
            return (st.st_mtime, st.st_size)
        except OSError:
            return None

    def _read(self, realpath):
        try:
            with open(realpath, errors="replace") as f:
                return f.read()
        except OSError:
            return None

    def _changed(self, e):
        """True if the file on disk no longer matches the (mtime, size) recorded
        at read time — a bash/edit/write/touch since the read."""
        sig = self._stat(e.realpath)
        if sig is None:
            return True
        return sig != (e.mtime, e.size)

    def current(self, realpath):
        """True iff the file is in context AND unchanged on disk since it was read."""
        e = self.entries.get(realpath)
        if not e or not e.in_context:
            return False
        return not self._changed(e)

    def status(self, realpath):
        """One of: 'unread', 'evicted' (read then dropped from context),
        'changed' (in context but stale on disk), 'current'."""
        e = self.entries.get(realpath)
        if not e or e.read_step is None:
            return "unread"
        if not e.in_context:
            return "evicted"
        if self._changed(e):
            return "changed"
        return "current"

    def covers(self, realpath, offset=1, limit=None):
        """True if the requested line range is already within the spans this file
        was read at (so a re-read would show nothing new)."""
        e = self.entries.get(realpath)
        if not e:
            return False
        if e.whole:
            return True
        content = self._read(realpath)
        if content is None:
            return False
        span = self._span(offset, limit, _nlines(content) or 1)
        if span is None:
            return e.whole
        s, en = span
        return any(a <= s and en <= b for a, b in e.spans)

    # ---- span math ----
    def _span(self, offset, limit, total):
        off = max(1, int(offset or 1))
        lim = DEFAULT_READ_LIMIT if limit in (None, "", 0) else max(1, int(limit))
        end = min(off + lim - 1, total)
        if off <= 1 and end >= total:
            return None          # whole file
        return (off, end)

    # ---- recording ----
    def record_read(self, realpath, step, offset=1, limit=None, content=None):
        """Record a successful read. A ranged read merges only ITS span; a
        whole-file read marks the file fully seen. Content is (re)read from disk
        for an authoritative sha1/diff baseline unless supplied."""
        if content is None:
            content = self._read(realpath)
        if content is None:
            return None
        span = self._span(offset, limit, _nlines(content) or 1)
        return self._ingest(realpath, step, content, span)

    def record_write(self, realpath, step, content=None):
        """Record a successful write_file/edit_file. The model authored the new
        content and holds it, so the whole file is marked seen + in context."""
        if content is None:
            content = self._read(realpath)
        if content is None:
            return None
        return self._ingest(realpath, step, content, None)

    def _ingest(self, realpath, step, content, span):
        e = self.entries.get(realpath)
        if e is None:
            e = Entry(realpath)
            self.entries[realpath] = e
        sig = self._stat(realpath)
        if sig:
            e.mtime, e.size = sig
        e.sha1 = _sha1(content)
        e.read_step = step
        e.in_context = True
        if span is None:
            e.whole = True
            e.spans = []
        elif not e.whole:
            e.spans = _merge(e.spans + [span])
        self._set_content(e, content)
        self._touch(e)
        self._enforce_total_cap()
        return e

    # ---- content cache (RAM caps) ----
    def _set_content(self, e, content):
        self._cached_bytes -= e._clen
        e._clen = 0
        e.content = None
        if content is not None and len(content) <= CONTENT_CAP:
            e.content = content
            e._clen = len(content)
            self._cached_bytes += e._clen

    def _drop_content(self, e):
        self._cached_bytes -= e._clen
        e._clen = 0
        e.content = None

    def _touch(self, e):
        self._tick += 1
        e._seq = self._tick

    def _enforce_total_cap(self):
        if self._cached_bytes <= TOTAL_CAP:
            return
        # LRU: drop cached content (keep metadata) from least-recently-touched first
        holders = [e for e in self.entries.values() if e.content is not None]
        holders.sort(key=lambda e: e._seq)
        for e in holders:
            if self._cached_bytes <= TOTAL_CAP:
                break
            self._drop_content(e)

    # ---- lifecycle ----
    def refresh(self):
        """os.stat every ledger path and flip changed entries stale. Bounded —
        one stat per tracked file. Catches bash/redirect mutations that changed
        mtime, so the next gate check sees the file as no longer current."""
        for e in self.entries.values():
            if self._changed(e):
                self.mark_mutated(e.realpath)

    def evict(self, realpath):
        """Compaction removed this file's observation from context: flip
        in_context False and drop cached content, but KEEP metadata so a later
        read is recognized as a re-read."""
        e = self.entries.get(realpath)
        if e:
            e.in_context = False
            e.obs_msg = None
            self._drop_content(e)

    def mark_mutated(self, realpath):
        """Explicitly poison an entry's stat baseline so current() reports it as
        changed until the next record — for a mutation the harness knows about
        but can't re-hash cheaply. Keeps cached content (the OLD version) so a
        re-read can still diff."""
        e = self.entries.get(realpath)
        if e:
            e.mtime = None

    def set_obs_msg(self, realpath, msg):
        """Attach the message dict that carries this read's observation, so
        compaction can detect (by identity) when it drops out of context."""
        e = self.entries.get(realpath)
        if e:
            e.obs_msg = msg

    # ---- diff-since-last-read ----
    def diff(self, realpath, name=None):
        """A capped unified diff between the cached content and the file on disk.
        Returns '' if content is byte-identical (a touch), a diff string if it
        changed, or None if there's no cached baseline to diff against."""
        e = self.entries.get(realpath)
        if not e or e.content is None:
            return None
        cur = self._read(realpath)
        if cur is None:
            return None
        if _sha1(cur) == e.sha1:
            return ""
        label = name or os.path.basename(realpath)
        lines = list(difflib.unified_diff(
            e.content.splitlines(), cur.splitlines(),
            fromfile=f"{label}@step{e.read_step}", tofile=f"{label}@now", lineterm=""))
        if len(lines) > DIFF_CAP:
            extra = len(lines) - DIFF_CAP
            lines = lines[:DIFF_CAP] + [f"... (diff truncated, {extra} more lines)"]
        return "\n".join(lines)

    # ---- reporting ----
    def held(self):
        """[(realpath, nlines_or_None)] for every file still in context — used by
        the compaction summary's 'files you have read and still hold' line."""
        out = []
        for e in self.entries.values():
            if e.in_context:
                out.append((e.realpath, _nlines(e.content) if e.content is not None else None))
        return out
