import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import plot_event_markers as event
import plot_index_vs_raw as plot
from lilia.custom_marker_quality_io import load_custom_marker_quality_table
from lilia.provenance import file_sha256
from test_csv_contract_regression import write_recording


class CustomMarkerQualityRegression(unittest.TestCase):
    def test_between_window_gaps_break_lines_without_dropping_valid_points(self):
        for gap_us in (0, 8000, 10000000):
            with self.subTest(gap_us=gap_us), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root/'recording.csv'
                time = np.arange(5000, dtype=np.int64)*2000
                time[2500:] += gap_us
                values = np.sin(np.arange(5000)/17.)*20
                write_recording(source, ''.join(f'{t},{v:.8f}\n' for t, v in zip(time, values)),
                                offset=0, channels='value', rate=500)
                seen = []

                def inspect(fig, path, **kwargs):
                    if str(path).endswith('.png'):
                        for ax, label in [(fig.axes[0], 'Focus absolute'),
                                          (fig.axes[0], 'Δ vs 5s pre-marker'),
                                          (fig.axes[4], 'Raw legacy quality')]:
                            line = next(line for line in ax.lines if line.get_label() == label)
                            data = np.asarray(line.get_ydata(), dtype=float)
                            self.assertEqual(len(data), 3 if gap_us else 2)
                            self.assertEqual(int(np.isfinite(data).sum()), 2)
                            if gap_us:
                                self.assertTrue(np.isnan(data[1]))
                            self.assertEqual(line.get_marker(), '.')
                        raw = np.asarray(fig.axes[5].lines[0].get_ydata())
                        self.assertEqual(int(np.isfinite(raw).sum()), 5000)
                        self.assertEqual(len(raw), 5001 if gap_us else 5000)
                        seen.append(True)

                with patch.object(Figure, 'savefig', inspect), \
                     patch.object(event, 'get_eeg_quality_index_v2_parametric',
                                  side_effect=lambda data, **kw: {'overall': np.full(data.shape[0], .7)}), \
                     contextlib.redirect_stdout(io.StringIO()):
                    plot.plot_custom_markers(str(source), 'Gap', 1, ['07:59:55'], '1970-01-01',
                        5, 30, str(root/'plot.png'), str(root/'summary.csv'))
                self.assertEqual(seen, [True])
                frame, _ = load_custom_marker_quality_table(root/'summary_quality_windows.csv', source)
                self.assertEqual(frame.crosses_gap.tolist(), [False, False])
                self.assertEqual(frame.quality_good.tolist(), [True, True])

    def test_direct_helper_preserves_two_result_api_and_adds_raw_audit(self):
        t = np.arange(5000, dtype=np.int64) * 2000
        x = np.sin(np.arange(5000) / 17.)[:, None].astype(np.float32)
        old_times, old_scores = event.compute_quality_windowed(t, x)
        times, scores, rows = event.compute_quality_windowed(t, x, return_audit=True)
        self.assertEqual(old_times, times)
        np.testing.assert_array_equal(old_scores, scores)
        self.assertEqual(len(rows), 2)
        self.assertEqual([r['window_start_idx'] for r in rows], [0, 2500])
        self.assertTrue(all(r['quality_diagnostics']['request']['stage'] == 'raw' for r in rows))
        self.assertTrue(all(r['quality_diagnostics']['state'] == 'valid' for r in rows))

    def test_old_scorer_unavailable_and_error_propagation(self):
        t = np.arange(1000, dtype=np.int64) * 5000
        x = np.sin(np.arange(1000) / 13.)[:, None]
        def old_scorer(data, fs, params):
            return {'overall': np.array([.7])}
        with patch.object(event, 'get_eeg_quality_index_v2_parametric', side_effect=old_scorer) as scorer:
            _, quality, rows = event.compute_quality_windowed(t, x, win_sec=5, fs=200,
                                                               return_audit=True)
        self.assertEqual(scorer.call_count, 1)
        np.testing.assert_array_equal(quality, [[.7]])
        self.assertEqual(rows[0]['quality_diagnostics']['state'], 'unavailable')
        with patch.object(event, 'get_eeg_quality_index_v2_parametric', side_effect=RuntimeError('score broke')):
            with self.assertRaisesRegex(RuntimeError, 'score broke'):
                event.compute_quality_windowed(t, x, win_sec=5, fs=200, return_audit=True)

    def test_source_reader_rejects_rehashed_window_and_diagnostic_forgery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'recording.csv'
            samples = np.sin(np.arange(5000) / 21.) * 16.
            rows = ''.join(f'{i * 2000},{value:.8f}\n' for i, value in enumerate(samples))
            write_recording(source, rows, offset=0, channels='value', rate=500)
            with contextlib.redirect_stdout(io.StringIO()):
                plot.plot_custom_markers(str(source), 'Demo', 1, ['08:00:05'], '1970-01-01',
                    5, 5, str(root / 'out.png'), str(root / 'summary.csv'))
            table = root / 'summary_quality_windows.csv'
            frame, meta = load_custom_marker_quality_table(table, source)
            self.assertEqual(len(frame), 2)
            self.assertTrue((root / 'out.png').exists())
            self.assertEqual(set(frame.quality_stage), {'raw'})
            np.testing.assert_array_equal(frame.quality_good.to_numpy(),
                np.isfinite(frame.quality_ch1.to_numpy()) &
                (frame.quality_ch1.to_numpy() >= meta['parameters']['quality_threshold']))
            changes = (
                lambda f, m: f.__setitem__('window_start_idx', [1, 2500]),
                lambda f, m: f.__setitem__('quality_ch1', f.quality_ch1 + .1),
                lambda f, m: f.__setitem__('crosses_gap', [True, False]),
                lambda f, m: m['parameters'].__setitem__('quality_threshold', -1.0),
                lambda f, m: m['quality_audit'][0]['quality_diagnostics']['request'].__setitem__('stage', 'filtered'),
            )
            for i, change in enumerate(changes):
                forged = root / f'forged_{i}.csv'
                shutil.copyfile(table, forged)
                saved = json.loads(Path(str(table) + '.meta.json').read_text())
                altered = pd.read_csv(forged, float_precision='round_trip')
                change(altered, saved)
                altered.to_csv(forged, index=False)
                saved['table_sha256'] = file_sha256(forged)
                Path(str(forged) + '.meta.json').write_text(json.dumps(saved) + '\n')
                with self.subTest(forgery=i), self.assertRaises(ValueError):
                    load_custom_marker_quality_table(forged, source)


if __name__ == '__main__':
    unittest.main()
