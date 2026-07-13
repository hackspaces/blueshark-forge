"""H13 — XES/OCEL-compatible event export. Stdlib, offline."""
import csv
import io
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge import export as E                                       # noqa: E402
from forge.export import EXPORT_SCHEMA_VERSION                      # noqa: E402


def _lifecycle(run, turn, aid, kind, outcome, parent=None, attempt=1, detail=""):
    return {"type": "action_lifecycle", "run_id": run, "turn_id": turn, "action_id": aid,
            "action_kind": kind, "parent_action_id": parent, "attempt": attempt,
            "stage": outcome, "outcome": outcome, "detail": detail,
            "timestamps": {"requested": 1.0, "terminal": 2.0}}


_RECORDS = [
    {"type": "meta", "model": "qwen3-coder:30b", "contract": {"goal": "add a flag"}},
    _lifecycle("run-1", "t1", "a1", "read_file", "succeeded"),
    _lifecycle("run-1", "t1", "a2", "edit_file", "failed"),
    _lifecycle("run-1", "t1", "a3", "edit_file", "succeeded", parent="a2", attempt=2),  # a retry of a2
    {"type": "user", "text": "not an event"},
]


class TestNormalization(unittest.TestCase):
    def test_one_event_per_lifecycle_ignoring_other_records(self):
        events = E.to_events(_RECORDS)
        self.assertEqual(len(events), 3)                       # only the 3 lifecycles
        self.assertEqual({e["activity"] for e in events}, {"read_file", "edit_file"})

    def test_round_trips_core_identities(self):
        events = E.to_events(_RECORDS)
        a3 = next(e for e in events if e["action_id"] == "a3")
        self.assertEqual(a3["case_id"], "run-1")
        self.assertEqual(a3["turn_id"], "t1")
        self.assertEqual(a3["parent_action_id"], "a2")         # causal parent preserved
        self.assertEqual(a3["attempt"], 2)                     # retry attempt preserved
        self.assertEqual(a3["resource"], "qwen3-coder:30b")    # model as resource
        self.assertEqual(a3["task"], "add a flag")

    def test_parallel_actions_preserve_lifecycle_and_causality(self):
        # two concurrent actions in the same run keep distinct identities + their outcomes
        events = E.to_events(_RECORDS)
        by_id = {e["action_id"]: e for e in events}
        self.assertEqual(by_id["a2"]["outcome"], "failed")
        self.assertEqual(by_id["a3"]["outcome"], "succeeded")
        self.assertEqual(by_id["a3"]["parent_action_id"], "a2")

    def test_schema_version_is_explicit(self):
        for e in E.to_events(_RECORDS):
            self.assertEqual(e["schema_version"], EXPORT_SCHEMA_VERSION)


class TestRedaction(unittest.TestCase):
    def test_known_secret_shapes_are_masked(self):
        for secret in ("sk-abcdefghijklmnopqrstuvwx1234",
                       "ghp_0123456789abcdefghijklmnopqrstuvwxyz",
                       "AKIA0123456789ABCDEF"):
            out = E.redact(f"the key is {secret} ok")
            self.assertNotIn(secret, out)
            self.assertIn("<redacted:", out)

    def test_key_value_masks_the_value_keeps_the_key(self):
        out = E.redact('API_KEY=supersecretvalue123')
        self.assertNotIn("supersecretvalue123", out)
        self.assertIn("API_KEY", out)                          # the key name is kept
        self.assertIn("<redacted:", out)

    def test_redaction_is_deterministic(self):
        self.assertEqual(E.redact("sk-abcdefghijklmnopqrstuvwx1234"),
                         E.redact("sk-abcdefghijklmnopqrstuvwx1234"))

    def test_review_leak_cases_are_all_redacted(self):
        # every leak the H13 review found must now be masked.
        leaks = {
            "git clone https://admin:hunter2pass@github.com/x/y.git": "hunter2pass",
            "DATABASE_URL=postgres://dbuser:s3cr3tpw@db:5432/prod": "s3cr3tpw",
            "token eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.dozjgNryP4J3abcDEF": "dozjgNryP4J3abcDEF",
            "Authorization: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcDEFghiJKL": "abcDEFghiJKL",
            "AWS_SECRET_KEY=x1y2z": "x1y2z",                       # short value
            'password = "correct horse battery staple"': "correct horse battery staple",  # quoted+spaces
            'TOKEN=abc"defghij': 'abc"defghij',                    # embedded quote
            "-----BEGIN PGP PRIVATE KEY BLOCK-----\nabc123def\n-----END PGP PRIVATE KEY BLOCK-----": "abc123def",
            "value wJalrXUtnFEMI0K7MDENGbPxRfiCYEXAMPLEKEY here": "wJalrXUtnFEMI0K7MDENGbPxRfiCYEXAMPLEKEY",
        }
        for text, secret in leaks.items():
            self.assertNotIn(secret, E.redact(text), f"leaked: {secret}")

    def test_ordinary_text_is_left_alone(self):
        # redaction must not mangle normal command output / identifiers.
        for ok in ("read_file src/forge/agent.py", "ran 12 tests, all passed", "edit_file: 3 hunks"):
            self.assertEqual(E.redact(ok), ok)


class TestCsvSafetyAndEmpty(unittest.TestCase):
    def test_csv_formula_injection_is_neutralized(self):
        recs = [{"type": "meta", "model": "m", "contract": {"goal": "g"}},
                _lifecycle("r", "t", "a1", "bash", "succeeded", detail="=SUM(A1)+cmd|calc")]
        rows = list(csv.DictReader(io.StringIO(E.to_csv(E.to_events(recs)))))
        self.assertTrue(rows[0]["detail"].startswith("'="))       # prefixed → inert as a formula

    def test_empty_and_no_lifecycle_sessions_yield_empty_exports(self):
        self.assertEqual(E.to_events([]), [])
        self.assertEqual(E.to_events([{"type": "user", "text": "hi"}]), [])
        for fmt in ("csv", "json", "ocel"):
            self.assertIsInstance(E.export([], fmt), str)          # no crash
        self.assertEqual(json.loads(E.to_json([]))["events"], [])
        self.assertEqual(len(list(csv.DictReader(io.StringIO(E.to_csv([]))))), 0)  # header only

    def test_export_is_deterministic(self):
        self.assertEqual(E.export(_RECORDS, "ocel"), E.export(_RECORDS, "ocel"))

    def test_no_raw_secret_reaches_any_export(self):
        recs = [{"type": "meta", "model": "m", "contract": {"goal": "g"}},
                _lifecycle("r", "t", "a1", "bash", "succeeded",
                           detail="ran with token ghp_0123456789abcdefghijklmnopqrstuvwxyz")]
        for fmt in ("csv", "json", "ocel"):
            self.assertNotIn("ghp_0123456789abcdefghijklmnopqrstuvwxyz", E.export(recs, fmt))


class TestFormats(unittest.TestCase):
    def test_csv_has_header_and_a_row_per_event(self):
        rows = list(csv.DictReader(io.StringIO(E.to_csv(E.to_events(_RECORDS)))))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["schema_version"], str(EXPORT_SCHEMA_VERSION))

    def test_json_is_versioned_and_parseable(self):
        doc = json.loads(E.to_json(E.to_events(_RECORDS)))
        self.assertEqual(doc["schema_version"], EXPORT_SCHEMA_VERSION)
        self.assertEqual(len(doc["events"]), 3)

    def test_ocel_has_run_and_action_objects_and_relationships(self):
        doc = json.loads(E.to_ocel(E.to_events(_RECORDS)))
        self.assertEqual(doc["ocel:version"], "2.0")
        types = {o["type"] for o in doc["objects"]}
        self.assertEqual(types, {"run", "action"})
        # every event references its run and action objects
        for ev in doc["events"]:
            quals = {rel["qualifier"] for rel in ev["relationships"]}
            self.assertEqual(quals, {"run", "action"})
        self.assertIn("a2", [o["id"] for o in doc["objects"]])   # the failed action is an object


if __name__ == "__main__":
    unittest.main()
