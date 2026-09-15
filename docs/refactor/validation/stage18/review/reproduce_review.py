"""Read-only product review probes; all generated data stays in --out."""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import spectral_entropy as spectral
import quality_check
from lilia import quality
from lilia.entropy_io import load_joint_mi_table
from lilia.provenance import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    out = parser.parse_args().out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('joint_mi_review_tests', ROOT / 'tests/test_joint_mi_windows_regression.py')
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    case = tests.JointMIWindowTests()
    case.setUp()
    try:
        case.root = out
        source = case.recording()
        # Reject every MI series window, then independently recompute the
        # unmasked population from the actual float32 source values.
        with patch.object(spectral, '_eeg_quality_v2', return_value={'overall': np.zeros(2)}):
            table = case.run_cli(source, '--no-bandpass')
        frame, metadata = load_joint_mi_table(table, source)
        summary = pd.read_csv(out / 'out/recording_joint_mi_ch1_ch2_summary.csv').iloc[0]
        signal = case.x.astype(np.float32)
        expected = spectral.compute_joint_probability(signal[:, 0], signal[:, 1], bins=8, binning='quantile')
        population = dict(total_windows=len(frame), accepted_windows=int(frame.quality_valid.sum()),
            finite_window_mi=int(frame.joint_mi.notna().sum()), population_samples=int(summary.n_samples),
            population_quality_state=summary.population_quality_state,
            population_mi=float(summary.mutual_information_bits),
            independent_unmasked_mi=expected['mutual_information'])
        assert population['accepted_windows'] == population['finite_window_mi'] == 0
        assert population['population_samples'] == len(signal)
        population['mi_abs_difference'] = abs(population['population_mi'] - population['independent_unmasked_mi'])
        assert population['mi_abs_difference'] < 1e-14
        assert population['population_quality_state'] == 'scored'

        # A reader should reject a state that contradicts quality_enabled, even
        # after the table hash is recomputed. Preserve the original artifacts.
        changed = out / 'tampered_joint_mi.csv'
        tampered = frame.copy()
        tampered['quality_state'] = 'disabled'
        tampered.to_csv(changed, index=False)
        changed_metadata = {**metadata, 'table_sha256': file_sha256(changed)}
        Path(str(changed) + '.meta.json').write_text(json.dumps(changed_metadata, indent=2))
        loaded, loaded_meta = load_joint_mi_table(changed, source)
        reader = dict(accepted_inconsistent_state=True, quality_enabled=loaded_meta['parameters']['quality_enabled'],
                      quality_state=loaded.quality_state.unique().tolist())

        params = dict(flat_weight=0., spectrum_weight=1., kurtosis_weight=0., corr_weight=0.)
        def fallback(data, fs, params):
            with patch.object(quality.sp_signal, 'welch', side_effect=RuntimeError('review fallback')):
                return quality.get_eeg_quality_index_v2_parametric(data, fs=fs, params=params)
        t = np.arange(1000, dtype=np.int64) * 2000
        data = np.random.default_rng(18111).normal(size=(1000, 4)).astype(np.float32)
        with patch.object(spectral, '_eeg_quality_v2', side_effect=fallback), \
                patch.object(spectral, '_QUALITY_PARAMS', params):
            epochs, audit = spectral.collect_clean_epochs(t, data, data, 0, 2000000, 0,
                                                          fs=500, epoch_sec=1., quality_threshold=.5)
        lost = dict(accepted_epochs=len(epochs),
                    quality_states=[row['quality_state'] for row in audit['windows']],
                    saved_fields=sorted(audit['windows'][0]),
                    source_diagnostic_valid=fallback(data[:500].T.astype(float), 500, params)['valid'].tolist())
        assert len(epochs) == 2 and not any(lost['source_diagnostic_valid'])
        assert 'quality_diagnostics' not in audit['windows'][0]

        with patch.object(quality_check, 'get_eeg_quality_index_v2_parametric', return_value={'overall': np.full(4, np.nan)}):
            rows = quality_check.analyze_quality_windows(t, data, fs=500, win_sec=2., params=params)
        flags = quality_check.flag_anomalies(rows)
        anomalies = dict(nonfinite_quality=not np.isfinite(flags[0]['qmed']), reasons=flags[0]['reasons'],
                         severity=flags[0]['severity'])
        assert anomalies['nonfinite_quality'] and anomalies['reasons'] == []

        tracked = set(subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines())
        ignored = subprocess.check_output(['git', 'ls-files', '--others', '--ignored', '--exclude-standard',
            'docs/refactor/validation/stage18/raw_callers', 'docs/refactor/validation/stage18/tflite_callers'],
            cwd=ROOT, text=True).splitlines()
        missing_visual = []
        for scope in ('raw_callers', 'tflite_callers'):
            manifest_path = Path('docs/refactor/validation/stage18') / scope / 'visual_evidence.json'
            manifest = json.loads((ROOT / manifest_path).read_text())
            for row in manifest['files']:
                path = str(manifest_path.parent / row['path'])
                if path not in tracked:
                    missing_visual.append(path)
        findings = dict(head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                        population_label=population, reader_state=reader, entropy_fallback=lost,
                        anomaly_nan=anomalies,
                        committed_artifacts=dict(ignored_count=len(ignored), ignored_files=ignored,
                            missing_visual_manifest_targets=missing_visual))
        (out / 'findings.json').write_text(json.dumps(findings, indent=2, ensure_ascii=False) + '\n')
        print(json.dumps({k: v for k, v in findings.items() if k != 'committed_artifacts'}, ensure_ascii=False, indent=2))
        print('Ignored artifacts:', len(ignored), '; missing committed visual targets:', len(missing_visual))
        print('Review findings reproduced:', out / 'findings.json')
    finally:
        with contextlib.redirect_stdout(io.StringIO()):
            case.doCleanups()


if __name__ == '__main__':
    main()
