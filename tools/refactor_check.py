"""Quiet, repeatable refactor checks. See docs/refactor/WORKFLOW.md."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE_DIRS = ('lilia', 'tests', 'tools', 'plot_index_vs_raw_bundle', 'signal_quality_package')
# Only known readers; a manifest cannot select arbitrary Python code.
READERS = {
    'band_entropy': ('entropy_io', 'load_entropy_table', False),
    'joint_mi': ('entropy_io', 'load_joint_mi_table', False),
    'denoised_joint_mi': ('neural_io', 'load_denoised_joint_mi_table', True),
    'tflite_qeeg': ('tflite_io', 'load_tflite_table', True),
    'state_band_entropy': ('state_entropy_io', 'load_state_entropy_table', False),
    'event_marker_bp': ('event_qeeg_io', 'load_event_qeeg_table', False),
    'event_marker_tflite': ('event_qeeg_io', 'load_event_qeeg_table', True),
    'hardy2_band_indices': ('hardy2_io', 'load_hardy2_table', False),
    'subject_comparison_bp': ('subject_comparison_io', 'load_comparison_table', False),
    'subject_comparison_tflite': ('subject_comparison_io', 'load_comparison_table', True),
    'meditation_bp': ('meditation_io', 'load_meditation_table', False),
    'meditation_tflite': ('meditation_io', 'load_meditation_table', True),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def python_files():
    paths = set(ROOT.glob('*.py'))
    for directory in SOURCE_DIRS:
        paths.update((ROOT / directory).rglob('*.py'))
    return sorted(paths)


def snapshot(inputs=()):
    """Conservative fingerprint, not a dependency inference or result cache."""
    paths = set(python_files())
    paths.update((ROOT / 'tests' / 'fixtures').rglob('*'))
    for pattern in ('*.pth', '*.tflite', '*requirement*.txt', '*.toml', '*.sh'):
        paths.update(ROOT.glob(pattern))
    files = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths) if p.is_file()}
    extra = {str(Path(p).resolve()): sha256(p) for p in sorted(set(map(str, inputs)))}
    packages = sorted((d.metadata.get('Name', ''), d.version)
                      for d in importlib.metadata.distributions())
    environment = {key: os.environ.get(key) for key in
                   ('PYTHONPATH', 'PYTHONHASHSEED', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                    'OPENBLAS_NUM_THREADS', 'CUDA_VISIBLE_DEVICES', 'TF_DETERMINISTIC_OPS')}
    return {'files': files, 'inputs': extra, 'packages': [list(p) for p in packages],
            'python': sys.version, 'executable': sys.executable,
            'platform': platform.platform(), 'environment': environment}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_command(name, command, outdir, tests=False):
    log = outdir / f'{name}.log'
    env = {**os.environ, 'MPLBACKEND': 'Agg', 'MPLCONFIGDIR': str(outdir / 'mpl')}
    with log.open('w') as stream:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    row = {'name': name, 'command': command, 'exit_code': result.returncode,
           'passed': result.returncode == 0, 'log': str(log)}
    if tests:
        output = log.read_text(errors='replace')
        counts = re.findall(r'^Ran (\d+) tests? in ', output, re.MULTILINE)
        row['tests'] = int(counts[-1]) if counts else 0
        skips = re.findall(r'\bskipped=(\d+)', output)
        row['skipped'] = int(skips[-1]) if skips else 0
        row['passed'] = row['passed'] and row['tests'] > 0
    print(f"{'PASS' if row['passed'] else 'FAIL'} {name}"
          + (f" ({row['tests']} tests, {row['skipped']} skipped)" if tests else '')
          + (f' — {log}' if not row['passed'] else ''), flush=True)
    return row


def check(args, outdir):
    # Reject a typo before running anything; unittest discovery alone accepts zero tests.
    selected = []
    for name in args.tests or []:
        path = (ROOT / name).resolve()
        if path.parent != ROOT / 'tests' or not path.is_file() or not path.name.startswith('test_'):
            raise ValueError(f'Expected an existing tests/test_*.py: {name}')
        selected.append(path.name)
    before = snapshot(args.input)
    rows = []
    if args.full or args.static:
        paths = python_files()
        log = outdir / 'compile.log'
        errors = []
        for path in paths:
            try:
                compile(path.read_bytes(), str(path), 'exec')
            except (SyntaxError, ValueError) as error:
                errors.append(f'{path}: {error}')
        log.write_text('\n'.join(errors) + f'\n{len(paths)} Python files checked\n')
        rows.append({'name': 'compile', 'passed': not errors, 'files': len(paths), 'log': str(log)})
        print(f"{'FAIL' if errors else 'PASS'} compile ({len(paths)} files)", flush=True)
        for name, command in (
            ('pyflakes', [sys.executable, '-m', 'pyflakes', *map(str, paths)]),
            ('bundle', [sys.executable, 'build_bundles.py', '--check']),
            ('diff', ['git', 'diff', '--check', 'HEAD']),
        ):
            rows.append(run_command(name, command, outdir))
    patterns = ['test_*.py'] if args.full else selected
    for index, pattern in enumerate(patterns):
        rows.append(run_command(f'tests-{index + 1}',
                                [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
                                 '-p', pattern, '-v'], outdir, tests=True))
    after = snapshot(args.input)
    return {'scope': 'full' if args.full else 'static' if args.static else selected,
            'checks': rows, 'snapshot': before, 'inputs_unchanged': before == after,
            'passed': bool(rows) and all(r['passed'] for r in rows) and before == after}


def compare_arrays(actual, expected, *, rtol, atol, equal_nan=False):
    import numpy as np

    if not (np.isfinite(rtol) and np.isfinite(atol) and rtol >= 0 and atol >= 0):
        raise ValueError('Tolerances must be finite and nonnegative')
    errors = {}
    with np.load(actual, allow_pickle=False) as a, np.load(expected, allow_pickle=False) as b:
        if not a.files or set(a.files) != set(b.files):
            raise ValueError('NPZ keys differ or are empty')
        for key in sorted(a.files):
            x, y = a[key], b[key]
            if x.shape != y.shape or x.size == 0:
                raise ValueError(f'{key}: shape differs or array is empty')
            if x.dtype.kind not in 'biuf' or y.dtype.kind not in 'biuf':
                raise ValueError(f'{key}: only real numeric arrays are supported')
            # Integer timestamps/indices must never pass through tolerant float comparison.
            if x.dtype.kind in 'biu' or y.dtype.kind in 'biu':
                if x.dtype.kind != y.dtype.kind or not np.array_equal(x, y):
                    raise ValueError(f'{key}: integer values/types differ')
                errors[key] = 0.0
                continue
            np.testing.assert_allclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan)
            finite = np.isfinite(x) & np.isfinite(y)
            errors[key] = float(np.max(np.abs(x[finite] - y[finite]))) if finite.any() else None
    return errors


def evidence(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != 1 or set(manifest) - {
            'schema_version', 'files', 'tables', 'comparisons'}:
        raise ValueError('Unsupported manifest schema or fields')
    inputs = {manifest_path}

    def resolve(value):
        path = (manifest_path.parent / value).resolve()
        inputs.add(path)
        return path

    rows = []
    for entry in manifest.get('files', []):
        path = resolve(entry['path'])
        if sha256(path) != entry['sha256']:
            raise ValueError(f'Hash mismatch: {path}')
        rows.append({'name': 'hash', 'path': str(path), 'passed': True})
    for entry in manifest.get('tables', []):
        path, raw = resolve(entry['path']), resolve(entry['raw_csv'])
        sidecar = resolve(str(path) + '.meta.json')
        meta = json.loads(sidecar.read_text())
        kind = meta['kind']
        if kind != entry['kind'] or kind not in READERS:
            raise ValueError(f'Unsupported or unexpected table kind: {kind}')
        module, function, needs_model = READERS[kind]
        if kind == 'event_marker_bp' and meta['parameters'].get('scope') == 'event_zoom':
            module, function = 'event_zoom_io', 'load_zoom_table'
        reader = getattr(importlib.import_module('lilia.' + module), function)
        kwargs = {'model_path': resolve(entry['model_path'])} if needs_model else {}
        frame, _ = reader(path, raw, **kwargs)
        rows.append({'name': 'table', 'path': str(path), 'kind': kind,
                     'rows': len(frame), 'passed': True})
    for entry in manifest.get('comparisons', []):
        actual, expected = resolve(entry['actual']), resolve(entry['expected'])
        errors = compare_arrays(actual, expected, rtol=entry['rtol'], atol=entry['atol'],
                                equal_nan=entry.get('equal_nan', False))
        rows.append({'name': 'comparison', 'actual': str(actual), 'expected': str(expected),
                     'rtol': entry['rtol'], 'atol': entry['atol'],
                     'equal_nan': entry.get('equal_nan', False),
                     'max_abs_errors': errors, 'passed': True})
    if not rows:
        raise ValueError('Manifest has no checks')
    return rows, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    checks = sub.add_parser('check', help='Run selected tests or full stage checks')
    group = checks.add_mutually_exclusive_group(required=True)
    group.add_argument('--tests', nargs='+', help='Existing tests/test_*.py paths')
    group.add_argument('--full', action='store_true')
    group.add_argument('--static', action='store_true')
    checks.add_argument('--input', action='append', default=[], help='Additional fixture/input to fingerprint')
    proof = sub.add_parser('evidence', help='Verify hashes, source-aware tables and numeric NPZ pairs')
    proof.add_argument('manifest', type=Path)
    for command in (checks, proof):
        command.add_argument('--out', type=Path, help='New log directory; default unique /tmp directory')
    status = sub.add_parser('status', help='Check whether a saved check report still matches inputs')
    status.add_argument('report', type=Path)
    args = parser.parse_args()
    if args.action == 'status':
        try:
            report = json.loads(args.report.read_text())
            saved = report.get('snapshot')
            current = snapshot(saved['inputs']) if saved else None
            valid = bool(saved and report.get('passed') and saved == current)
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(json.dumps({'matching_passed_check': False, 'reason': str(error)}))
            return 1
        print(json.dumps({'matching_passed_check': valid, 'scope': report.get('scope'),
                          'recorded_at': report.get('recorded_at'),
                          'note': 'Fingerprint match only; no tests were run.'}))
        return 0 if valid else 1
    outdir = args.out.resolve() if args.out else Path(tempfile.mkdtemp(prefix='lilia-refactor-'))
    outdir.mkdir(parents=True, exist_ok=True)
    if any(outdir.iterdir()):
        parser.error('--out must be empty; preserve earlier logs by using a new directory')
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['MPLCONFIGDIR'] = str(outdir / 'mpl')
    report = {'schema_version': 1, 'recorded_at': datetime.now(timezone.utc).isoformat()}
    try:
        if args.action == 'check':
            report.update(check(args, outdir))
        else:
            # Evidence reruns are explicit; unlike check reports they are not reusable via status.
            # Source-aware readers verify the table contracts, not inference or scientific validity.
            log = outdir / 'evidence.log'
            import contextlib
            with log.open('w') as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                rows, inputs = evidence(args.manifest)
            report.update(scope='evidence', checks=rows, passed=True,
                          inputs={str(p): sha256(p) for p in sorted(inputs)}, log=str(log))
            print(f'PASS evidence ({len(rows)} checks)', flush=True)
    except Exception as error:
        import traceback
        log = outdir / 'error.log'
        log.write_text(traceback.format_exc())
        report.update(passed=False, error=f'{type(error).__name__}: {error}', log=str(log))
        print(f'FAIL — {log}', flush=True)
    save_json(outdir / 'summary.json', report)
    print(f"{'PASS' if report['passed'] else 'FAIL'} summary: {outdir / 'summary.json'}", flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
