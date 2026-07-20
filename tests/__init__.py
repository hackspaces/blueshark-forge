"""Make tests/ a package so `from tests import _hermetic` resolves everywhere,
and apply the real-~/.forge protection for package-form invocations
(`python3 -m unittest tests.<module>`) before any sibling module loads.
Bare `discover -s tests` never imports this file — that is why every test
module imports tests._hermetic itself; see tests/_hermetic.py.
"""
from tests import _hermetic  # noqa: F401
