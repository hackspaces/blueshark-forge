"""H11 — full-fidelity workspace fixtures: capture, restore, fidelity, safety."""
import os
import stat as statmod
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge import fixture as FX                                    # noqa: E402
from forge.receipt import workspace_digest                        # noqa: E402


def _write(d, rel, content, binary=False):
    fp = os.path.join(d, rel)
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    mode = "wb" if binary else "w"
    with open(fp, mode) as f:
        f.write(content)


def _reasons(fx):
    return {e["path"]: e["reason"] for e in fx["excluded"]}


class TestCaptureRestore(unittest.TestCase):
    def test_capture_then_restore_reproduces_the_workspace(self):
        # acceptance: the environment is reproduced — restored content matches the source.
        src = tempfile.mkdtemp()
        _write(src, "app.py", "x = 1\n")
        _write(src, "pkg/mod.py", "y = 2\n")
        _write(src, "data.bin", b"\x00\x01\x02\xff", binary=True)
        fx = FX.capture(src)
        dest = tempfile.mkdtemp()
        FX.restore(fx, dest)
        # every captured file is byte-identical, and the two trees share a digest
        self.assertEqual(workspace_digest(src), workspace_digest(dest))
        with open(os.path.join(dest, "data.bin"), "rb") as f:
            self.assertEqual(f.read(), b"\x00\x01\x02\xff")   # binary via base64 round-trips

    def test_fixture_is_content_addressed(self):
        a, b = tempfile.mkdtemp(), tempfile.mkdtemp()
        for d in (a, b):
            _write(d, "x.py", "same\n")
        self.assertEqual(FX.capture(a)["digest"], FX.capture(b)["digest"])
        _write(b, "x.py", "different\n")
        self.assertNotEqual(FX.capture(a)["digest"], FX.capture(b)["digest"])

    def test_digest_framing_defeats_content_crafted_collisions(self):
        # 'a' containing "x\nb:utf-8:y" must NOT collide with files a="x", b="y" —
        # the naive join-with-separators digest did exactly that.
        one = {"a": {"encoding": "utf-8", "content": "x\nb:utf-8:y"}}
        two = {"a": {"encoding": "utf-8", "content": "x"},
               "b": {"encoding": "utf-8", "content": "y"}}
        self.assertNotEqual(FX._digest(one), FX._digest(two))

    def test_executable_bit_round_trips(self):
        src = tempfile.mkdtemp()
        _write(src, "build.sh", "#!/bin/sh\necho ok\n")
        os.chmod(os.path.join(src, "build.sh"), 0o755)
        fx = FX.capture(src)
        self.assertTrue(fx["files"]["build.sh"].get("exec"))     # captured
        dest = tempfile.mkdtemp()
        FX.restore(fx, dest)
        mode = os.stat(os.path.join(dest, "build.sh")).st_mode
        self.assertTrue(mode & statmod.S_IXUSR)                  # ./build.sh replays


class TestSecretAndSizeExclusion(unittest.TestCase):
    def test_secret_files_are_excluded_never_archived(self):
        src = tempfile.mkdtemp()
        _write(src, "app.py", "code\n")
        _write(src, ".env", "SECRET_TOKEN=supersecret\n")
        _write(src, "deploy.pem", "-----BEGIN PRIVATE KEY-----\nabc\n")
        _write(src, ".ssh/id_rsa", "PRIVATE\n")
        fx = FX.capture(src)
        # the secrets' CONTENTS never enter the fixture
        blob = str(fx["files"])
        self.assertNotIn("supersecret", blob)
        self.assertNotIn("BEGIN PRIVATE KEY", blob)
        self.assertNotIn("app.py", str(fx["excluded"]))       # normal code is kept
        self.assertIn("app.py", fx["files"])
        # and each exclusion is an explicit fidelity limitation (the .ssh dir is pruned whole)
        limits = "\n".join(FX.fidelity_limitations(fx))
        for secret in (".env", "deploy.pem", ".ssh"):
            self.assertIn(secret, limits)

    def test_secret_matching_is_case_insensitive(self):
        # macOS's default filesystem is case-insensitive: Deploy.PEM IS a .pem key.
        src = tempfile.mkdtemp()
        _write(src, ".ENV", "T=CASELEAK\n")
        _write(src, "Deploy.PEM", "PEMLEAK\n")
        _write(src, "ID_RSA", "RSALEAK\n")
        fx = FX.capture(src)
        blob = str(fx["files"])
        for leak in ("CASELEAK", "PEMLEAK", "RSALEAK"):
            self.assertNotIn(leak, blob)
        self.assertEqual(len(fx["excluded"]), 3)              # each an honest limitation

    def test_dot_env_suffix_variants_are_secrets(self):
        # docker-compose env_file convention: prod.env / secrets.env carry credentials.
        src = tempfile.mkdtemp()
        _write(src, "prod.env", "TOKEN=ENVLEAK\n")
        fx = FX.capture(src)
        self.assertNotIn("ENVLEAK", str(fx["files"]))
        self.assertEqual(_reasons(fx).get("prod.env"), "secret")

    def test_env_template_is_not_treated_as_a_secret(self):
        src = tempfile.mkdtemp()
        _write(src, ".env.example", "SECRET_TOKEN=changeme\n")
        fx = FX.capture(src)
        self.assertIn(".env.example", fx["files"])            # a template is safe to capture

    def test_symlinked_file_is_never_read_through(self):
        # a link named innocently can point at real credentials outside the tree.
        outside = tempfile.mkdtemp()
        _write(outside, "credentials", "AWS_SECRET=LINKLEAK\n")
        src = tempfile.mkdtemp()
        os.symlink(os.path.join(outside, "credentials"), os.path.join(src, "awsconf"))
        fx = FX.capture(src)
        self.assertNotIn("LINKLEAK", str(fx["files"]))        # contents never archived
        self.assertEqual(_reasons(fx).get("awsconf"), "symlink")

    def test_symlinked_dir_is_an_honest_limitation_not_a_silent_gap(self):
        outside = tempfile.mkdtemp()
        _write(outside, "part.csv", "1,2\n")
        src = tempfile.mkdtemp()
        os.symlink(outside, os.path.join(src, "data"))
        fx = FX.capture(src)
        self.assertEqual(fx["files"], {})                     # not followed…
        self.assertEqual(_reasons(fx).get("data"), "symlink-dir")   # …but never silent

    @unittest.skipUnless(hasattr(os, "mkfifo"), "no mkfifo on this platform")
    def test_fifo_does_not_hang_capture(self):
        src = tempfile.mkdtemp()
        _write(src, "app.py", "ok\n")
        os.mkfifo(os.path.join(src, "pipe"))
        fx = FX.capture(src)                                  # must return, not block
        self.assertIn("app.py", fx["files"])
        self.assertEqual(_reasons(fx).get("pipe"), "not-a-regular-file")

    def test_large_files_are_excluded_as_a_fidelity_limitation(self):
        src = tempfile.mkdtemp()
        _write(src, "big.dat", "x" * 10)
        _write(src, "small.py", "ok\n")
        fx = FX.capture(src, max_file_bytes=5)
        self.assertIn("small.py", fx["files"])
        self.assertNotIn("big.dat", fx["files"])
        self.assertTrue(any("too-large" in line for line in FX.fidelity_limitations(fx)))

    def test_aggregate_budget_bounds_the_capture(self):
        src = tempfile.mkdtemp()
        for i in range(5):
            _write(src, f"f{i}.py", "x\n")
        fx = FX.capture(src, max_files=2)
        self.assertEqual(len(fx["files"]), 2)                 # capture stops at the budget
        self.assertTrue(any(e["reason"] == "capture-budget" for e in fx["excluded"]))


class TestSafety(unittest.TestCase):
    def test_restore_never_touches_the_original_workspace(self):
        src = tempfile.mkdtemp()
        _write(src, "app.py", "original\n")
        fx = FX.capture(src)
        # mutate the fixture and restore ELSEWHERE — the source must be untouched
        fx["files"]["app.py"]["content"] = "tampered\n"
        dest = tempfile.mkdtemp()
        FX.restore(fx, dest)
        with open(os.path.join(src, "app.py")) as f:
            self.assertEqual(f.read(), "original\n")          # original unchanged
        with open(os.path.join(dest, "app.py")) as f:
            self.assertEqual(f.read(), "tampered\n")

    def test_restore_refuses_a_path_escaping_dest(self):
        dest = tempfile.mkdtemp()
        malicious = {"version": 1, "files": {"../../etc/pwned": {"encoding": "utf-8", "content": "x"}}}
        FX.restore(malicious, dest)
        self.assertFalse(os.path.exists("/etc/pwned"))        # traversal refused


class TestReplayIntegration(unittest.TestCase):
    def _recs(self, cwd):
        return [{"type": "meta", "model": "m", "cwd": cwd, "mode": "auto"},
                {"type": "user", "text": "do it"},
                {"type": "model", "raw": '{"action":"say","message":"ok"}', "tier": 0, "prompt_tokens": 1}]

    def test_write_fixture_captures_workspace_excluding_secrets(self):
        import json
        from forge import replay as R
        ws = tempfile.mkdtemp()
        _write(ws, "app.py", "code\n")
        _write(ws, ".env", "TOKEN=secret123\n")
        fxdir = tempfile.mkdtemp()
        orig_rec, orig_dir = R._records_for, R.fixtures_dir
        R._records_for = lambda sid: self._recs(ws)
        R.fixtures_dir = lambda: fxdir
        try:
            path = R.write_fixture("sid", "wtest")
            wpath = path[:-len(".jsonl")] + ".workspace.json"
            self.assertTrue(os.path.exists(wpath))            # companion workspace fixture written
            with open(wpath) as f:
                wf = json.load(f)
            self.assertIn("app.py", wf["files"])
            self.assertNotIn("secret123", json.dumps(wf["files"]))   # secret not archived
            self.assertTrue(any(".env" in line for line in FX.fidelity_limitations(wf)))
        finally:
            R._records_for, R.fixtures_dir = orig_rec, orig_dir

    def test_rerecording_removes_a_stale_companion(self):
        # a companion from a PRIOR recording must never pair with new raws.
        from forge import replay as R
        ws = tempfile.mkdtemp()
        _write(ws, "app.py", "v1\n")
        fxdir = tempfile.mkdtemp()
        orig_rec, orig_dir = R._records_for, R.fixtures_dir
        R.fixtures_dir = lambda: fxdir
        try:
            R._records_for = lambda sid: self._recs(ws)
            path = R.write_fixture("sid", "stale")
            wpath = path[:-len(".jsonl")] + ".workspace.json"
            self.assertTrue(os.path.exists(wpath))
            # re-record with a GONE cwd → the old companion must be removed, not kept
            R._records_for = lambda sid: self._recs("/nonexistent-forge-h11")
            R.write_fixture("sid", "stale")
            self.assertFalse(os.path.exists(wpath))           # stale companion gone
        finally:
            R._records_for, R.fixtures_dir = orig_rec, orig_dir

    def test_replay_fixture_cleans_up_the_restored_workspace(self):
        import tempfile as tf
        from forge import replay as R
        ws = tempfile.mkdtemp()
        _write(ws, "app.py", "code\n")
        fxdir = tempfile.mkdtemp()
        orig_rec, orig_dir = R._records_for, R.fixtures_dir
        R._records_for = lambda sid: self._recs(ws)
        R.fixtures_dir = lambda: fxdir
        made = []
        orig_mkdtemp = tf.mkdtemp

        def spy(**kw):
            d = orig_mkdtemp(**kw)
            if kw.get("prefix") == "forge-wsfx-":
                made.append(d)
            return d
        tf.mkdtemp = spy
        try:
            path = R.write_fixture("sid", "clean")
            R.replay_fixture(path)
            self.assertTrue(made)                             # a workspace WAS restored…
            self.assertFalse(os.path.exists(made[0]))         # …and removed afterwards
        finally:
            tf.mkdtemp = orig_mkdtemp
            R._records_for, R.fixtures_dir = orig_rec, orig_dir


if __name__ == "__main__":
    unittest.main()
