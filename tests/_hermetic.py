"""Process-wide protection of the developer's real ~/.forge — imported by EVERY
test module (`from tests import _hermetic`), so it fires no matter how the suite
is invoked: bare `discover -s tests` (imports modules top-level, never the
package __init__), `discover -t .`, `python3 -m unittest tests.<module>`, or a
single module run from inside tests/.

Why not the package __init__ alone: discover's top_level_dir DEFAULTS to the
start dir, so `discover -s tests` — the canonical command in CLAUDE.md and CI —
imports test modules as top-level names and tests/__init__.py never executes.
That was found the hard way: earlier suite runs leaked exemplar/malformed
entries under test backend names into the real ~/.forge ("script": 172
malformed strikes), and the P5.6 cold-start pin then made head-layout
assertions machine-dependent — a test passed in the full suite and failed in
any subset that skipped test_exemplars.py's import-time redirect.

This module is SAFETY (never read or write the real home). Modules that also
need a FRESH store for isolation (test_exemplars.py, test_profile.py, ...)
keep their own module-level re-redirects on top — that's a different job.
"""
import os
import tempfile

from forge import config as _config
from forge import exemplars as _exemplars
from forge import profile as _profile

_SUITE_HOME = tempfile.mkdtemp(prefix="forge-suite-home-")
_exemplars.EXEMPLAR_DIR = os.path.join(_SUITE_HOME, "exemplars")
_profile.PROFILE_DIR = os.path.join(_SUITE_HOME, "profile")
# a path that does not exist → config.load() serves pure defaults; a developer's
# tuned ~/.forge/config.json (e.g. stuck_threshold) must not skew assertions,
# and a test that WRITES config must not touch the real home.
_config.PATH = os.path.join(_SUITE_HOME, "config.json")
