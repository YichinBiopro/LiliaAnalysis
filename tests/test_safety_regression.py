import argparse
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from lilia.provenance import write_goertzel_metadata, validate_goertzel_cache
from lilia.quality_policy import valid_goertzel_rows

ROOT = Path(__file__).resolve().parents[1]


class SafetyRegressionTests(unittest.TestCase):
    def test_cache_rejects_changes_and_legacy_without_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, table = Path(tmp) / 'source.csv', Path(tmp) / 'table.csv'
            source.write_text('raw samples')
            table.write_text('analysis samples')
            with self.assertRaisesRegex(ValueError, 'metadata missing'):
                validate_goertzel_cache(table, source, {})
            write_goertzel_metadata(table, source, {'fs': 500, 'win_sec': 5})
            validate_goertzel_cache(table, source, {'fs': 500})
            with self.assertRaisesRegex(ValueError, 'changed'):
                validate_goertzel_cache(table, source, {'win_sec': 2})
            source.write_text('changed samples')
            with self.assertRaisesRegex(ValueError, 'changed'):
                validate_goertzel_cache(table, source, {'fs': 500})
            source.write_text('raw samples')
            table.write_text('modified output')
            with self.assertRaisesRegex(ValueError, 'changed'):
                validate_goertzel_cache(table, source, {})

    def test_quality_policy_rejects_nonfinite_and_hard_artifacts(self):
        df = pd.DataFrame({'quality': [1.] * 5, 'quality_final': [0.8, 0.8, 0.8, np.nan, 0.8],
                           'goertzel_db': [1, 2, np.inf, 4, 5],
                           'time_s': [1, 2, 3, 4, np.nan], 'artifact_hard_clip': [0, 1, 0, 0, 0]})
        np.testing.assert_array_equal(valid_goertzel_rows(df), [True, False, False, False, False])
        np.testing.assert_array_equal(valid_goertzel_rows(df, exclude_hard=False),
                                      [True, True, False, True, False])

    def test_goertzel_windows_after_gap_preserve_indices_and_wall_time(self):
        import plot_goertzel_vs_raw as plot
        t = np.r_[np.arange(3500) * 2000, 20000000 + np.arange(3500) * 2000]
        data = np.sin(np.arange(7000)[:, None] / 7)
        with patch.object(plot, 'get_eeg_quality_index_v2_parametric', return_value={'overall': np.ones(1)}):
            result = plot._compute_window_metrics(t, data, data, 500, 1, 60, 5, 1, {},
                                                  1950, 0.12, 1000, 1, 80)
        np.testing.assert_array_equal(result.window_start_idx, [0, 500, 1000, 3500, 4000, 4500])
        np.testing.assert_allclose(result.time_s, [2.5, 3.5, 4.5, 22.5, 23.5, 24.5])

    def test_mi_onsets_exclude_nonparticipants(self):
        import joint_mi
        import spectral_entropy as spectral
        events = [('restricted', '12:00', 1, ['Hardy']), ('all', '12:01', 1, None)]
        t = np.arange(100000) * 2000
        hhmm = lambda value: {'12:00': 50000000, '12:01': 110000000}[value]
        with patch.object(joint_mi, 'EVENTS', events), patch.object(joint_mi, 'hhmm_to_us', side_effect=hhmm):
            self.assertEqual(joint_mi.resolve_events(t, len(t), 500, 500, subject='Ann'), [('all', 55000)])
        args = argparse.Namespace(ibrain_events=True, subject='Ann', fs=500)
        with patch.object(spectral, '_IBRAIN_EVENTS', events), patch.object(spectral, '_hhmm_to_us', side_effect=hhmm):
            self.assertEqual(spectral._resolve_event_onsets(args, t, len(t)), [55000])
            self.assertEqual(spectral.compute_event_pre_onset_joint_mi(
                np.ones(1000), np.ones(1000), 0, subject='Ann', n_surrogates=0), [])

    def test_validation_shell_propagates_each_failure_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / 'python'
            fake.write_text('#!/bin/bash\n'
                            'if [[ "$FAIL_STAGE" == compile && "$2" == py_compile ]]; then exit 9; fi\n'
                            'if [[ "$FAIL_STAGE" == help && "$2" == --help ]]; then exit 8; fi\n'
                            'if [[ "$FAIL_STAGE" == tests && "$2" == unittest ]]; then exit 7; fi\n'
                            'exit 0\n')
            fake.chmod(0o755)
            for stage in ['compile', 'help', 'tests']:
                with self.subTest(stage=stage):
                    env = dict(os.environ, PATH=f'{root}:{os.environ["PATH"]}', FAIL_STAGE=stage,
                               LILIA_TEST_LOG=str(root / 'log'))
                    result = subprocess.run(['bash', str(ROOT / 'validate_signal_processing.sh')], env=env,
                                            capture_output=True, text=True, cwd=tmp)
                    self.assertEqual(result.returncode, 1, result.stdout)
