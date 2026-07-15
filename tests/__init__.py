"""Process-wide hermetic redirects for machine-global state — applied on ANY
import through the `tests` package, i.e. before any test in any subset runs.

Why this exists: these redirects used to live at import time inside individual
test modules (test_exemplars.py, test_profile.py, ...), which only protects runs
that happen to import THOSE modules. Full discovery imports everything, so the
full suite was hermetic — but `python3 -m unittest tests.test_bench` alone read
the real ~/.forge, where prior suite runs had already leaked exemplar/malformed
entries under test backend names ("script": 172 malformed strikes), and the P5.6
cold-start head-pin fired two extra head messages: the same test then passed in
the full suite and failed in any subset. Hermeticity must not depend on which
sibling modules got imported.

The per-module redirects are kept (harmless duplicates) so running a module
directly from inside tests/ (no package import) stays protected where it
already was.
"""
import os
import tempfile

from forge import config as _config
from forge import exemplars as _exemplars
from forge import profile as _profile

_SUITE_HOME = tempfile.mkdtemp(prefix="forge-suite-home-")
_exemplars.EXEMPLAR_DIR = os.path.join(_SUITE_HOME, "exemplars")
_profile.PROFILE_DIR = os.path.join(_SUITE_HOME, "profile")
# a path that does not exist → config.load() serves pure defaults; a test that
# tunes e.g. stuck_threshold on the developer's real machine must not leak into
# assertions (and a test that WRITES config must not touch the real home).
_config.PATH = os.path.join(_SUITE_HOME, "config.json")
