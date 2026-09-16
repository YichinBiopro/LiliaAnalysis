"""M1 harness failures must not disappear behind equal legacy/current outputs."""
from pathlib import Path
import unittest

from tools.freeze_baseline_entries import canonical, check_outcome, check_policy_witnesses, marker_clock


class BaselineEntryFreezeTests(unittest.TestCase):
    def test_unexpected_failure_and_missing_expected_failure_are_rejected(self):
        check_outcome({'name': 'ok'}, None)
        check_outcome({'name': 'missing', 'error_contains': 'no usable'}, 'no usable windows')
        for case, error in [({'name': 'ok'}, 'no usable'),
                            ({'name': 'missing', 'error_contains': 'no usable'}, None),
                            ({'name': 'missing', 'error_contains': 'no usable'}, 'different bug')]:
            with self.assertRaises(AssertionError):
                check_outcome(case, error)

    def test_canonical_retains_source_model_table_and_selection_evidence(self):
        value = {'artifacts': {'plot.png': 'renderer-specific'}, 'source_id': 'source',
                 'table_sha256': 'table', 'model_sha256': 'model',
                 'selection': [1, 3], 'error': 'audit: /run/case/analysis.json'}
        actual = canonical(value, Path('/run/case'))
        self.assertNotIn('artifacts', actual)
        for key in ('source_id', 'table_sha256', 'model_sha256', 'selection'):
            self.assertEqual(actual[key], value[key])
        self.assertEqual(actual['error'], 'audit: <output>/analysis.json')

    def test_equal_but_wrong_boundary_decisions_are_rejected(self):
        comparison = {'products': {'markers_summary.csv': {
            'baseline_windows': [6, 6, 5, 0], 'post_windows': [5, 6, 5, 0]}}}
        check_policy_witnesses({'name': 'gap_markers'}, comparison)
        comparison['products']['markers_summary.csv']['baseline_windows'][1] = 7
        with self.assertRaises(AssertionError):
            check_policy_witnesses({'name': 'gap_markers'}, comparison)

    def test_recorded_marker_clock_preserves_six_decimal_places(self):
        self.assertEqual(marker_clock(1781749707900242), ('2026-06-18', '10:28:27.900242'))
