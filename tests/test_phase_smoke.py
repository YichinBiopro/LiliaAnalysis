import importlib
import os
import subprocess
import sys
import unittest

from lilia.pathing import get_project_root


class PhaseSmokeTests(unittest.TestCase):
    def test_project_root_exists(self):
        root = get_project_root()
        self.assertTrue(os.path.isdir(root))

    def test_signal_quality_package_imports(self):
        mod = importlib.import_module("signal_quality_package.eeg_quality_v2")
        self.assertTrue(hasattr(mod, "get_eeg_quality_index_v2_parametric"))

    def test_core_quality_imports(self):
        importlib.import_module("signal_quality_package.quality")
        importlib.import_module("eeg_quality_v2")
        importlib.import_module("lilia.quality")

    def test_help_smoke_data_analysis(self):
        proc = subprocess.run(
            [sys.executable, "data_analysis.py", "--help"],
            cwd=get_project_root(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("--group", proc.stdout)

    def test_import_convert_to_tflite(self):
        mod = importlib.import_module("convert_to_tflite")
        self.assertTrue(hasattr(mod, "MODEL_PATH"))

    def test_help_smoke_goertzel_sampling(self):
        proc = subprocess.run(
            [sys.executable, "sample_segments_by_goertzel_db.py", "--help"],
            cwd=get_project_root(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("--mode", proc.stdout)

    def test_help_smoke_goertzel_resample_wrapper(self):
        proc = subprocess.run(
            [sys.executable, "resample_clean_goertzel_samples.py", "--help"],
            cwd=get_project_root(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)

    def test_help_smoke_goertzel_histograms(self):
        proc = subprocess.run(
            [sys.executable, "plot_goertzel_histograms.py", "--help"],
            cwd=get_project_root(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)

    def test_help_smoke_goertzel_summary(self):
        proc = subprocess.run(
            [sys.executable, "summarize_goertzel_distribution.py", "--help"],
            cwd=get_project_root(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)

    def test_help_smoke_plot_raw_eeg(self):
        proc = subprocess.run(
            [sys.executable, "plot_raw_eeg.py", "--help"],
            cwd=get_project_root(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)

    def test_help_smoke_quality_check(self):
        proc = subprocess.run(
            [sys.executable, "quality_check.py", "--help"],
            cwd=get_project_root(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)

    def test_help_smoke_plot_goertzel_vs_raw(self):
        proc = subprocess.run(
            [sys.executable, "plot_goertzel_vs_raw.py", "--help"],
            cwd=get_project_root(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)

    def test_help_smoke_filter_hard_artifacts(self):
        proc = subprocess.run(
            [sys.executable, "filter_and_plot_hard_artifacts.py", "--help"],
            cwd=get_project_root(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)

    def test_help_smoke_plot_tyy_meditation(self):
        proc = subprocess.run(
            [sys.executable, "plot_tyy_meditation.py", "--help"],
            cwd=get_project_root(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)


if __name__ == "__main__":
    unittest.main()
