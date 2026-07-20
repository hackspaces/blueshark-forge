"""H06 — declared tool effects + capability grants. Stdlib, offline.

Acceptance: every built-in tool is declared; unknown tools/effects fail closed;
the capability grant is a deterministic function of authority (never the model);
denials name the missing capability; the declaration agrees with enforcement.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests import _hermetic  # noqa: E402,F401 — never touch the real ~/.forge

from forge import effects as E                                   # noqa: E402
from forge.authority import AuthorityLevel, AuthorityPolicy, ACTION_AUTHORITY  # noqa: E402


class TestDeclarations(unittest.TestCase):
    def test_every_builtin_tool_is_declared(self):
        # the effect table and the authority level map cover exactly the same tools
        self.assertEqual(set(E.TOOLS), set(ACTION_AUTHORITY))

    def test_declared_min_authority_agrees_with_enforcement(self):
        # the declared floor equals the level authority enforces, and a session AT
        # that level is permitted. The absolute allow/deny boundary — including denial
        # BELOW the floor — is pinned by test_absolute_authorization_matrix.
        for name, spec in E.TOOLS.items():
            self.assertEqual(spec.min_authority, ACTION_AUTHORITY[name], name)
            at = AuthorityPolicy(spec.min_authority.name.lower())
            self.assertTrue(at.evaluate({"action": name, "target": "peer", "message": "x"}).allowed, name)

    def test_absolute_authorization_matrix(self):
        # Pin the ABSOLUTE allow/deny boundary so lowering any floor — even in BOTH
        # effects.TOOLS and ACTION_AUTHORITY together — fails loudly. The agreement
        # test only proves the two tables match, not that the shared floor is correct.
        levels = ["observe", "contribute", "operator", "admin"]
        rank = {name: i for i, name in enumerate(levels)}
        expected_floor = {  # the minimum authority at which the tool's core capability is granted
            "read_file": "observe", "list_files": "observe", "grep": "observe",
            "glob": "observe", "say": "observe",
            "write_file": "contribute", "edit_file": "contribute", "run_tests": "contribute",
            "bash": "operator", "fleet_send": "operator",   # send; the observe roster is tested separately
        }
        self.assertEqual(set(expected_floor), set(E.TOOLS))   # every tool pinned
        for tool, floor in expected_floor.items():
            for lvl in levels:
                action = {"action": tool, "command": "echo hi", "target": "peer", "message": "m"}
                allowed = AuthorityPolicy(lvl).evaluate(action).allowed
                self.assertEqual(allowed, rank[lvl] >= rank[floor], f"{tool}@{lvl}")

    def test_bash_dangerous_command_still_escalates_to_admin(self):
        # the declared operator floor must NOT let the shell classifier be bypassed —
        # a dangerous command still requires admin (defense in depth over the declaration).
        danger = AuthorityPolicy("operator").evaluate({"action": "bash", "command": "sudo rm -rf /"})
        self.assertFalse(danger.allowed)
        self.assertEqual(danger.required, AuthorityLevel.ADMIN)
        # a benign command stays at the declared operator floor
        self.assertTrue(AuthorityPolicy("operator").evaluate({"action": "bash", "command": "ls"}).allowed)

    def test_every_tool_has_a_nonempty_or_intentional_effect(self):
        for name, spec in E.TOOLS.items():
            # only `say` is declared with no side effects
            if name == "say":
                self.assertEqual(spec.effects, E.Effect.NONE)
            else:
                self.assertNotEqual(spec.effects, E.Effect.NONE, name)


class TestFailClosed(unittest.TestCase):
    def test_unknown_tool_needs_admin_and_has_every_effect(self):
        spec = E.spec_for("definitely_not_a_tool")
        self.assertEqual(spec.min_authority, AuthorityLevel.ADMIN)
        self.assertEqual(spec.effects, E.ALL_EFFECTS)

    def test_unknown_tool_is_denied_below_admin(self):
        d = AuthorityPolicy("operator").evaluate({"action": "quantum_rm"})
        self.assertFalse(d.allowed)
        self.assertEqual(d.required, AuthorityLevel.ADMIN)


class TestCapabilityGrant(unittest.TestCase):
    def test_grant_is_deterministic_and_model_independent(self):
        # granted_capabilities takes ONLY a level — same level, same grant, always.
        self.assertEqual(E.granted_capabilities(AuthorityLevel.CONTRIBUTE),
                         E.granted_capabilities(AuthorityLevel.CONTRIBUTE))

    def test_grant_is_monotonic_in_authority(self):
        levels = [AuthorityLevel.OBSERVE, AuthorityLevel.CONTRIBUTE,
                  AuthorityLevel.OPERATOR, AuthorityLevel.ADMIN]
        grants = [set(E.granted_capabilities(l)) for l in levels]
        for lo, hi in zip(grants, grants[1:]):
            self.assertTrue(lo <= hi)                    # a higher level never LOSES a capability
        self.assertIn("bash", grants[2])                 # operator gains bash
        self.assertNotIn("bash", grants[1])              # contribute does not

    def test_grant_matches_legal_actions_except_the_fleet_roster(self):
        for lvl in (AuthorityLevel.OBSERVE, AuthorityLevel.CONTRIBUTE,
                    AuthorityLevel.OPERATOR, AuthorityLevel.ADMIN):
            legal = set(AuthorityPolicy(lvl.name.lower()).legal_actions())
            grant = set(E.granted_capabilities(lvl))
            if lvl < AuthorityLevel.OPERATOR:
                # fleet_send stays in the grammar (read-only roster) but its SEND
                # capability is not granted until operator — the one documented gap.
                self.assertIn("fleet_send", legal)
                self.assertNotIn("fleet_send", grant)
                self.assertEqual(grant, legal - {"fleet_send"})
            else:
                self.assertEqual(grant, legal)


class TestDenialNamesCapability(unittest.TestCase):
    def test_denied_action_names_the_missing_capability(self):
        d = AuthorityPolicy("observe").evaluate({"action": "bash", "command": "echo hi"})
        self.assertFalse(d.allowed)
        self.assertIn("run processes", d.effects)
        self.assertIn("write files", d.effects)
        self.assertIn("run processes", d.reason)        # named in the human message too
        self.assertIn("requires operator authority", d.reason)
        self.assertEqual(d.to_dict()["effects"], d.effects)   # structured field survives serialization

    def test_describe(self):
        self.assertEqual(E.describe(E.Effect.NONE), "no side effects")
        s = E.describe(E.Effect.FS_WRITE | E.Effect.PROCESS)
        self.assertIn("write files", s)
        self.assertIn("run processes", s)
        self.assertNotIn("network", s)


if __name__ == "__main__":
    unittest.main()
