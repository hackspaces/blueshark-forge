"""Hardware-honest model tiers: CPU-only machines never get a swap-thrashing ladder."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests import _hermetic  # noqa: E402,F401 — never touch the real ~/.forge

from forge import setup as S                                       # noqa: E402

APPLE = {"os": "Darwin", "arch": "arm64", "ram_gb": 8, "chip": "Apple M2"}
WINTEL = {"os": "Windows", "arch": "AMD64", "ram_gb": 8, "chip": "Intel"}
LINTEL = {"os": "Linux", "arch": "x86_64", "ram_gb": 16, "chip": "AMD"}


def _max_b(ladder):
    """Largest parameter count (in B) named in a ladder, parsed from the ':Nb' tags."""
    import re
    sizes = []
    for m in ladder:
        for hit in re.findall(r":(\d+(?:\.\d+)?)b", m):
            sizes.append(float(hit))
    return max(sizes) if sizes else 0.0


class TestCpuOnlyTiers(unittest.TestCase):
    def test_8gb_cpu_only_never_gets_a_9b_rung(self):
        # the defect: TIERS' 8GB rung escalates into gemma2:9b — ~4 tok/s on shared
        # DDR and swap-thrash on a machine whose OS idles at 3-4GB.
        with mock.patch.object(S, "_is_accelerated", return_value=False):
            ladder, label = S.recommend(8, WINTEL)
        self.assertLessEqual(_max_b(ladder), 3.0)              # capped at ~3B
        self.assertIn("cpu", label)

    def test_16gb_cpu_only_tops_at_an_escalation_7b(self):
        with mock.patch.object(S, "_is_accelerated", return_value=False):
            ladder, _ = S.recommend(16, LINTEL)
        self.assertLessEqual(_max_b(ladder), 7.0)              # no 9B; 7B is the stuck-rung
        self.assertLessEqual(_max_b(ladder[:1]), 3.0)          # the DEFAULT rung stays small

    def test_apple_silicon_keeps_the_accelerated_ladder(self):
        ladder, _ = S.recommend(48, APPLE)                     # unified memory + Metal
        self.assertIn("qwen3-coder:30b", ladder)               # big MoE stays available

    def test_nvidia_dgpu_counts_as_accelerated(self):
        with mock.patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            self.assertTrue(S._is_accelerated(LINTEL))

    def test_no_fast_path_is_not_accelerated(self):
        with mock.patch("shutil.which", return_value=None):
            self.assertFalse(S._is_accelerated(WINTEL))

    def test_detected_vram_counts_as_accelerated(self):
        self.assertTrue(S._is_accelerated({"os": "Linux", "arch": "x86_64", "vram_gb": 80}))


class TestGpuDetection(unittest.TestCase):
    def test_detect_machine_reads_nvidia_gpus_and_vram(self):
        import subprocess
        real = subprocess.check_output

        def fake(cmd, *a, **k):
            if cmd and cmd[0] == "nvidia-smi":                 # a multi-GPU node
                return b"81920, NVIDIA A100-SXM4-80GB\n81920, NVIDIA A100-SXM4-80GB\n"
            return real(cmd, *a, **k)                          # real RAM/chip detection
        with mock.patch("subprocess.check_output", side_effect=fake):
            hw = S.detect_machine()
        self.assertEqual(hw["gpus"], 2)
        self.assertEqual(hw["vram_gb"], 160)                   # 2 × 80GB
        self.assertIn("A100", hw["gpu_name"])

    def test_no_nvidia_smi_leaves_zero_gpus(self):
        hw = S.detect_machine()                                # this machine has no nvidia-smi
        self.assertEqual(hw.get("gpus"), 0)
        self.assertEqual(hw.get("vram_gb"), 0)

    def test_recommend_without_hw_keeps_old_behavior(self):
        # back-compat: callers that pass no hw get the original table untouched.
        self.assertEqual(S.recommend(8), S.recommend(8, APPLE))


if __name__ == "__main__":
    unittest.main()
