import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from lilia.goertzel_sampling import (
    collect_candidates_per_subject,
    collect_matches_global,
    sample_rows,
)


def _write_minimal_merged_csv(path: Path) -> None:
    header = [
        "File Name, dummy",
        "Abs Time Offset[us], 0",
        "Sampling Rate[Hz], 500",
        "Columns, Time[us], Ch1",
    ]
    data = ["Time[us],Ch1", "0,0.0", "1000,0.1"]
    path.write_text("\n".join(header + data) + "\n", encoding="utf-8")


class GoertzelSamplingRegressionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "iBrainCenter"
        self.root.mkdir(parents=True, exist_ok=True)

        self.subj = self.root / "Demo(SN999)"
        self.subj.mkdir(parents=True, exist_ok=True)
        _write_minimal_merged_csv(self.subj / "merged.csv")

        df_ch1 = pd.DataFrame(
            {
                "time_s": [1.0, 2.0, 3.0, 4.0],
                "goertzel_db": [67.1, 66.0, 67.3, 70.0],
                "quality": [0.9, 0.3, 0.95, 0.99],
                "quality_final": [0.85, 0.2, 0.9, 0.99],
                "artifact_hard_clip": [0, 0, 1, 0],
                "window_start_idx": [0, 100, 200, 300],
                "window_end_idx": [50, 150, 250, 350],
            }
        )
        df_ch2 = pd.DataFrame(
            {
                "time_s": [1.5, 2.5],
                "goertzel_db": [67.0, 67.9],
                "quality": [0.8, 0.8],
            }
        )
        df_ch1.to_csv(self.subj / "index_vs_raw_goertzel_ch1_60Hz.csv", index=False)
        df_ch2.to_csv(self.subj / "index_vs_raw_goertzel_ch2_60Hz.csv", index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_collect_matches_global_filters_by_tolerance_and_quality(self):
        out = collect_matches_global(
            root=str(self.root),
            channels=[1, 2],
            stem="index_vs_raw_goertzel",
            target_freq=60.0,
            target_db=67.15,
            tol_db=0.25,
            quality_threshold=0.5,
        )
        self.assertEqual(len(out), 2)
        self.assertNotIn(3.0, out["time_s"].tolist())
        self.assertIn("window_start_idx", out.columns)
        self.assertEqual(out.loc[out["channel"] == 1, "quality"].iloc[0], 0.85)
        self.assertTrue(set(out["channel"].tolist()).issubset({1, 2}))

    def test_collect_candidates_per_subject_prefers_quality_final_and_excludes_hard_clip(self):
        out = collect_candidates_per_subject(
            root=str(self.root),
            subject="Demo(SN999)",
            channels=[1],
            stem="index_vs_raw_goertzel",
            target_freq=60.0,
            target_db=67.15,
            quality_threshold=0.5,
            fs=500.0,
        )
        self.assertEqual(len(out), 2)
        self.assertIn("distance_db", out.columns)
        self.assertTrue(np.all(out["artifact_hard_clip"].to_numpy(dtype=float) < 0.5))

    def test_sample_rows_is_reproducible(self):
        df = pd.DataFrame({"x": np.arange(20)})
        s1 = sample_rows(df, n=5, seed=7)
        s2 = sample_rows(df, n=5, seed=7)
        self.assertEqual(s1["x"].tolist(), s2["x"].tolist())


if __name__ == "__main__":
    unittest.main()
