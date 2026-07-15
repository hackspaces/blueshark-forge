"""The suite must never read or write the developer's real ~/.forge — see
tests/__init__.py, which redirects every machine-global store process-wide.

This test pins the property so it fails loudly if the package-level redirect is
ever removed: without it, a subset run (e.g. `python3 -m unittest
tests.test_bench`) reads real exemplar/profile/config state leaked by earlier
runs, and head-layout assertions become machine-dependent (a test that passed
in the full suite failed alone, because the P5.6 cold-start pin fired off real
`~/.forge/exemplars` entries recorded under test backend names).

Every invocation shape is protected because every test module imports
tests._hermetic itself (bare `discover -s tests` never runs the package
__init__ — its top_level_dir defaults to the start dir, so modules import as
top-level names).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests import _hermetic  # noqa: E402,F401 — never touch the real ~/.forge

from forge import config, exemplars, profile     # noqa: E402


class TestSuiteHermeticity(unittest.TestCase):
    def test_machine_global_state_is_redirected(self):
        home = os.path.realpath(os.path.expanduser("~/.forge"))
        for name, path in (("exemplars.EXEMPLAR_DIR", exemplars.EXEMPLAR_DIR),
                           ("profile.PROFILE_DIR", profile.PROFILE_DIR),
                           ("config.PATH", config.PATH)):
            self.assertFalse(os.path.realpath(path).startswith(home + os.sep),
                             f"{name} points into the real ~/.forge: {path}")
