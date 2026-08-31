import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from lilia.goertzel_distribution import (
    collect_group_data,
    common_db_edges,
    with_aggregates,
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


class GoertzelDistributionRegressionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "iBrainCenter"
        self.root.mkdir(parents=True, exist_ok=True)

        for idx, name in enumerate(["S1(SN001)", "S2(SN002)"]):
            sub = self.root / name
            sub.mkdir(parents=True, exist_ok=True)
            _write_minimal_merged_csv(sub / "merged.csv")
            df = pd.DataFrame(
                {
                    "goertzel_power": [1.0 + idx, 2.0 + idx, 3.0 + idx],
                    "goertzel_db": [66.5 + idx, 67.0 + idx, 68.0 + idx],
                    "quality": [0.9, 0.4, 0.95],
                    "quality_final": [0.8, 0.4, 0.95],
                    "artifact_hard_clip": [0, 0, 1],
                }
            )
            df.to_csv(sub / "index_vs_raw_goertzel_ch1_60Hz.csv", index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_collect_group_data_filters_rows(self):
        groups, missing = collect_group_data(
            root=str(self.root),
            channels=[1],
            stem="index_vs_raw_goertzel",
            target_freq=60.0,
            threshold=0.5,
            exclude_hard_artifact=True,
            require_power=True,
        )
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(missing), 0)
        self.assertTrue(all(g.n_total == 3 for g in groups))
        self.assertTrue(all(g.n_kept == 1 for g in groups))

    def test_with_aggregates_adds_channel_and_all_channels(self):
        groups, _ = collect_group_data(
            root=str(self.root),
            channels=[1],
            stem="index_vs_raw_goertzel",
            target_freq=60.0,
            threshold=0.5,
            exclude_hard_artifact=True,
            require_power=False,
        )
        all_groups = with_aggregates(groups, [1])
        tags = {(g.subject, g.channel) for g in all_groups}
        self.assertIn(("ALL_SUBJECTS", "ch1"), tags)
        self.assertIn(("ALL_SUBJECTS", "ALL_CHANNELS"), tags)

    def test_common_db_edges_shape(self):
        groups, _ = collect_group_data(
            root=str(self.root),
            channels=[1],
            stem="index_vs_raw_goertzel",
            target_freq=60.0,
            threshold=0.5,
            exclude_hard_artifact=True,
            require_power=False,
        )
        edges = common_db_edges(groups, bins=12)
        self.assertEqual(edges.shape[0], 13)
        self.assertTrue(np.all(np.diff(edges) > 0))


if __name__ == "__main__":
    unittest.main()
