import tempfile
import unittest
from pathlib import Path

from lilia.subject_paths import iter_group_merged_csvs, iter_subject_dirs


def _write_minimal_merged_csv(path: Path) -> None:
    lines = [
        "File Name, demo",
        "Abs Time Offset[us], 0",
        "Sampling Rate[Hz], 500",
        "Columns, Time[us], Ch1",
        "Time[us],Ch1",
        "0,0.0",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class SubjectDiscoveryRegressionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)

        ib = self.base / "iBrainCenter"
        yg = self.base / "YoGa"
        ib.mkdir(parents=True, exist_ok=True)
        yg.mkdir(parents=True, exist_ok=True)

        (ib / "A(SN001)").mkdir(parents=True, exist_ok=True)
        _write_minimal_merged_csv(ib / "A(SN001)" / "merged.csv")

        (ib / "B(SN002)").mkdir(parents=True, exist_ok=True)
        # B has no merged.csv and should not be discovered.

        (yg / "C(SN003)").mkdir(parents=True, exist_ok=True)
        _write_minimal_merged_csv(yg / "C(SN003)" / "merged.csv")

    def tearDown(self):
        self.tmp.cleanup()

    def test_iter_subject_dirs_filters_missing_merged(self):
        ib = self.base / "iBrainCenter"
        subjects = list(iter_subject_dirs(str(ib)))
        self.assertEqual(subjects, ["A(SN001)"])

    def test_iter_group_merged_csvs_returns_group_subject_path(self):
        rows = list(iter_group_merged_csvs(str(self.base), groups=("iBrainCenter", "YoGa")))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][0], "iBrainCenter")
        self.assertEqual(rows[0][1], "A(SN001)")
        self.assertTrue(rows[0][2].endswith("iBrainCenter/A(SN001)/merged.csv"))
        self.assertEqual(rows[1][0], "YoGa")
        self.assertEqual(rows[1][1], "C(SN003)")


if __name__ == "__main__":
    unittest.main()
