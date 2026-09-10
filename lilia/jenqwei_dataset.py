"""Segment-local dataset export and source-verifiable fragment contracts.

This prepares signal windows; it does not run a model or assign dataset labels.
Legacy *_200hz coordinates remain aliases of packed resampled coordinates at fs_out.
Every interval is half-open; raw fractional positions describe sample centres,
not the full filter support (which is the complete source segment).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter

from lilia.io import bandpass_filter, load_merged_csv
from lilia.provenance import file_sha256
from lilia.signal import _resampling_ratio, resample_segment_time_us, resample_with_time
from lilia.windowing import continuous_slices


@dataclass
class SegmentMeta:
    source_csv: str
    split_index: int
    split_count: int
    window_index_in_split: int
    window_count_in_split: int
    start_idx_200hz: int
    end_idx_200hz: int
    n_samples_200hz: int
    duration_sec: float
    out_csv: str
    source_id: str
    segment_id: int
    raw_start_idx: int
    raw_end_idx: int
    resampled_start_idx: int
    resampled_end_idx: int
    segment_local_start_idx: int
    segment_local_end_idx: int
    raw_fractional_first_idx: float
    raw_fractional_last_idx: float
    model_window_start_in_split: int
    model_window_count: int
    split_model_window_count: int
    fs_out: float


def parameters(n_splits=5, fs_in=500., fs_out=200., bp_low=.5, bp_high=45.,
               n_ch=4, tflite_win=400, allow_short_drop=False, one_window_per_file=False):
    for name, value in (('n_splits', n_splits), ('n_ch', n_ch), ('tflite_win', tflite_win)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    _resampling_ratio(fs_in, fs_out)
    if not (np.isfinite(bp_low) and np.isfinite(bp_high) and 0 < bp_low < bp_high < fs_in / 2):
        raise ValueError('Require finite 0 < bp_low < bp_high < fs_in / 2')
    if not isinstance(allow_short_drop, bool) or not isinstance(one_window_per_file, bool):
        raise ValueError('Drop and output mode must be boolean')
    return dict(n_splits=n_splits, fs_in=float(fs_in), fs_out=float(fs_out),
                bp_low=float(bp_low), bp_high=float(bp_high), n_ch=n_ch, tflite_win=tflite_win,
                allow_short_drop=allow_short_drop, one_window_per_file=one_window_per_file)


def split_bounds(n_samples, n_splits):
    if n_samples < 0 or n_splits <= 0:
        raise ValueError('Invalid split dimensions')
    return [(i * n_samples // n_splits, (i + 1) * n_samples // n_splits) for i in range(n_splits)]


def dataset_plan(time_us, params):
    """Plan equal splits BEFORE trimming, independently for each source segment."""
    p = parameters(**params)
    if np.asarray(time_us).dtype.kind not in 'iu':
        raise ValueError('Dataset time_us must be integer microseconds')
    up, dn = _resampling_ratio(p['fs_in'], p['fs_out'])
    sos = butter(4, [p['bp_low'], p['bp_high']], btype='bandpass', fs=p['fs_in'], output='sos')
    padlen = int(3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())))
    segments, fragments = [], []
    offset = 0
    for sid, sl in enumerate(continuous_slices(time_us, p['fs_in'])):
        n = ((sl.stop - sl.start) * up + dn - 1) // dn
        segment = dict(segment_id=sid, raw_start_idx=sl.start, raw_end_idx=sl.stop,
                       resampled_start_idx=offset, resampled_end_idx=offset + n, splits=[])
        for split_idx, (a, b) in enumerate(split_bounds(n, p['n_splits']), 1):
            reason = 'filter_too_short' if sl.stop - sl.start <= padlen else (
                'short_split' if b - a < p['tflite_win'] else '')
            keep = 0 if reason else (b - a) // p['tflite_win'] * p['tflite_win']
            nw = keep // p['tflite_win']
            segment['splits'].append(dict(split_index=split_idx, local_start_idx=a, local_end_idx=b,
                                          retained_end_idx=a + keep, retained_samples=keep,
                                          excluded_samples=b - a - keep,
                                          reason=reason or ('window_tail' if b - a != keep else 'none')))
            step = p['tflite_win'] if p['one_window_per_file'] else keep
            for start in range(a, a + keep, step or 1):
                end = min(start + step, a + keep)
                fragments.append(dict(
                    split_index=split_idx, split_count=p['n_splits'],
                    window_index_in_split=(start - a) // p['tflite_win'] + 1 if p['one_window_per_file'] else 1,
                    window_count_in_split=nw if p['one_window_per_file'] else 1,
                    start_idx_200hz=offset + start, end_idx_200hz=offset + end,
                    n_samples_200hz=end - start, duration_sec=(end - start) / p['fs_out'],
                    segment_id=sid, raw_start_idx=sl.start, raw_end_idx=sl.stop,
                    resampled_start_idx=offset + start, resampled_end_idx=offset + end,
                    segment_local_start_idx=start, segment_local_end_idx=end,
                    raw_fractional_first_idx=sl.start + start * dn / up,
                    raw_fractional_last_idx=sl.start + (end - 1) * dn / up,
                    model_window_start_in_split=(start - a) // p['tflite_win'],
                    model_window_count=(end - start) // p['tflite_win'], split_model_window_count=nw,
                    fs_out=p['fs_out']))
        segments.append(segment)
        offset += n
    return dict(policy='equal_splits_per_source_segment_then_window_trim', gap_factor=3.,
                filter_order=4, filter_padlen=padlen, filter_dtype='float32',
                resampling='scipy.signal.resample_poly default', index_space='packed_resampled_including_exclusions',
                quality_state='disabled', model_inference=False, segments=segments, fragments=fragments)


class DatasetPlanError(ValueError):
    def __init__(self, message, plan):
        super().__init__(message)
        self.plan = plan


def check_plan(plan, params):
    dropped = [(s['segment_id'], r['split_index'], r['reason']) for s in plan['segments']
               for r in s['splits'] if not r['retained_samples']]
    if dropped and not params['allow_short_drop']:
        raise DatasetPlanError(f'Cannot form windows; excluded (segment, split, reason): {dropped}', plan)
    if not plan['fragments']:
        raise DatasetPlanError('No complete dataset windows; all splits excluded', plan)


def source_contract(csv_path, params):
    time_us, data = load_merged_csv(csv_path)
    if data.shape[1] < params['n_ch']:
        raise ValueError(f'{csv_path} has {data.shape[1]} channels; requires {params["n_ch"]}')
    plan = dataset_plan(time_us, params)
    check_plan(plan, params)
    return time_us, data[:, :params['n_ch']], plan


def session_name(source, source_id, params):
    # Preserve continuous legacy filenames; allow_short_drop does not alter values.
    identity = {'source': str(Path(source).resolve()), 'source_sha256': source_id,
                **{k: v for k, v in params.items() if k != 'allow_short_drop'}}
    suffix = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    return f'{Path(source).stem}_{suffix}'


def fragment_name(session, fragment, params, segmented):
    sid = f'_seg{fragment["segment_id"]:04d}' if segmented else ''
    name = f'{session}{sid}_split{fragment["split_index"]:02d}-of-{params["n_splits"]:02d}'
    if params['one_window_per_file']:
        name += f'_w{fragment["window_index_in_split"]:04d}-of-{fragment["window_count_in_split"]:04d}'
    return name + f'_{int(params["fs_out"])}Hz_bp.csv'


def save_segment_csv(out_csv, time_us_seg, data_seg):
    columns = {'time_us': np.asarray(time_us_seg, dtype=np.int64)}
    columns.update({f'ch{i+1}': data_seg[:, i].astype(np.float32) for i in range(data_seg.shape[1])})
    pd.DataFrame(columns).to_csv(out_csv, index=False, mode='x')


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def process_source(csv_path, out_root, params):
    params = parameters(**params)
    source = str(Path(csv_path).resolve())
    source_id = file_sha256(source)
    t, raw, plan = source_contract(source, params)
    session = session_name(source, source_id, params)
    out = Path(out_root).resolve() / session
    out.mkdir(parents=True, exist_ok=False)
    rows, files = [], []
    for segment in plan['segments']:
        sid = segment['segment_id']
        fragments = [f for f in plan['fragments'] if f['segment_id'] == sid]
        if not fragments:
            continue
        a, b = segment['raw_start_idx'], segment['raw_end_idx']
        filtered = bandpass_filter(raw[a:b], fs=params['fs_in'], lo=params['bp_low'], hi=params['bp_high'])
        rt, values = resample_with_time(t[a:b], filtered, params['fs_in'], params['fs_out'])
        if not np.isfinite(values).all():
            raise ValueError(f'Non-finite processed signal in segment {sid}')
        for fragment in fragments:
            path = out / fragment_name(session, fragment, params, len(plan['segments']) > 1)
            start, end = fragment['segment_local_start_idx'], fragment['segment_local_end_idx']
            save_segment_csv(path, rt[start:end], values[start:end])
            row = SegmentMeta(source_csv=source, source_id=source_id, out_csv=str(path), **fragment)
            rows.append(row)
            files.append({'row': row.__dict__, 'sha256': file_sha256(path)})
    if file_sha256(source) != source_id:
        raise ValueError('Source changed during dataset generation')
    write_json(out / 'dataset.json', dict(schema_version=1, kind='jenqwei_dataset', source_csv=source,
                                         source_id=source_id, parameters=params, plan=plan, files=files))
    return rows


def _read_contract(audit_path, raw_csv):
    meta = json.loads(Path(audit_path).read_text())
    if meta.get('schema_version') != 1 or meta.get('kind') != 'jenqwei_dataset':
        raise ValueError('Unknown dataset schema or kind')
    if meta.get('source_id') != file_sha256(raw_csv) or meta.get('source_csv') != str(Path(raw_csv).resolve()):
        raise ValueError('Dataset source mismatch')
    params = parameters(**meta['parameters'])
    if params != meta['parameters']:
        raise ValueError('Dataset parameters mismatch')
    t, _, plan = source_contract(raw_csv, params)
    if meta.get('plan') != plan:
        raise ValueError('Dataset plan differs from source')
    session = session_name(raw_csv, meta['source_id'], params)
    expected = []
    for fragment in plan['fragments']:
        path = Path(audit_path).parent / fragment_name(session, fragment, params, len(plan['segments']) > 1)
        expected.append(dict(source_csv=str(Path(raw_csv).resolve()), source_id=meta['source_id'],
                             out_csv=str(path.resolve()), **fragment))
    if [item['row'] for item in meta['files']] != expected:
        raise ValueError('Dataset file coverage or mapping mismatch')
    actual = {str(p.resolve()) for p in Path(audit_path).parent.glob('*.csv')}
    if actual != {r['out_csv'] for r in expected}:
        raise ValueError('Missing or extra dataset fragments')
    return t, meta


def _read_fragment(path, t, meta, item):
    if file_sha256(path) != item['sha256']:
        raise ValueError('Dataset fragment hash mismatch')
    p, row = meta['parameters'], item['row']
    frame = pd.read_csv(path)
    if list(frame.columns) != ['time_us', *[f'ch{i+1}' for i in range(p['n_ch'])]]:
        raise ValueError('Dataset columns mismatch')
    rt = resample_segment_time_us(t[row['raw_start_idx']:row['raw_end_idx']], p['fs_in'], p['fs_out'])
    expected = rt[row['segment_local_start_idx']:row['segment_local_end_idx']]
    if frame.time_us.dtype.kind not in 'iu' or not np.array_equal(frame.time_us, expected):
        raise ValueError('Dataset integer timestamps or row count mismatch')
    if not np.isfinite(frame.iloc[:, 1:].to_numpy(dtype=np.float32)).all():
        raise ValueError('Non-finite dataset fragment')
    return frame


def load_fragment(path, raw_csv):
    """Validate full source plan/coverage and this CSV; does not re-filter values."""
    path = Path(path).resolve()
    t, meta = _read_contract(path.parent / 'dataset.json', raw_csv)
    matches = [item for item in meta['files'] if item['row']['out_csv'] == str(path)]
    if len(matches) != 1:
        raise ValueError('Fragment not uniquely listed in dataset')
    return _read_fragment(path, t, meta, matches[0]), meta


def write_manifest(out_root, rows, sources, params):
    root = Path(out_root).resolve()
    if not rows:
        raise ValueError('Cannot publish empty dataset')
    path = root / 'manifest.csv'
    if path.exists() or (root / 'manifest.meta.json').exists():
        raise FileExistsError('Dataset manifest already exists')
    sources = [str(Path(s).resolve()) for s in sources]
    if not sources or len(set(sources)) != len(sources):
        raise ValueError('Dataset sources must be nonempty and unique')
    audits = [root / session_name(source, file_sha256(source), params) / 'dataset.json' for source in sources]
    expected_rows = []
    for source, audit_path in zip(sources, audits):
        t, audit = _read_contract(audit_path, source)
        if audit['parameters'] != params:
            raise ValueError('Dataset audit parameters differ from manifest')
        for item in audit['files']:
            _read_fragment(item['row']['out_csv'], t, audit, item)
            expected_rows.append(item['row'])
    if [r.__dict__ for r in rows] != expected_rows:
        raise ValueError('Cannot publish incomplete or reordered dataset rows')
    pd.DataFrame([r.__dict__ for r in rows]).to_csv(path, index=False, mode='x')
    write_json(root / 'manifest.meta.json', dict(schema_version=1, kind='jenqwei_dataset_manifest',
               parameters=params, sources=[str(Path(s).resolve()) for s in sources],
               manifest_sha256=file_sha256(path),
               audits=[dict(path=str(a), sha256=file_sha256(a)) for a in audits]))
    return path


def load_manifest(path):
    """Read every fragment and validate manifest/source/file completeness."""
    path = Path(path).resolve()
    meta = json.loads(path.with_suffix('.meta.json').read_text())
    if (meta.get('schema_version') != 1 or meta.get('kind') != 'jenqwei_dataset_manifest'
            or file_sha256(path) != meta.get('manifest_sha256')):
        raise ValueError('Dataset manifest schema or hash mismatch')
    p = parameters(**meta['parameters'])
    sources = meta['sources']
    if not sources or len(set(sources)) != len(sources) or len(meta['audits']) != len(sources):
        raise ValueError('Dataset source coverage mismatch')
    rows = []
    expected_audits = set()
    for source, item in zip(sources, meta['audits']):
        expected = path.parent / session_name(source, file_sha256(source), p) / 'dataset.json'
        expected_audits.add(str(expected))
        if item['path'] != str(expected) or file_sha256(expected) != item['sha256']:
            raise ValueError('Dataset audit identity or hash mismatch')
        t, audit = _read_contract(expected, source)
        if audit['parameters'] != p:
            raise ValueError('Dataset audit parameters differ from manifest')
        for fragment in audit['files']:
            _read_fragment(fragment['row']['out_csv'], t, audit, fragment)
            rows.append(fragment['row'])
    if {str(a) for a in path.parent.glob('*/dataset.json')} != expected_audits:
        raise ValueError('Unexpected source audit')
    actual = pd.read_csv(path, float_precision='round_trip')
    expected = pd.DataFrame(rows)[list(SegmentMeta.__dataclass_fields__)]
    try:
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=True)
    except AssertionError as exc:
        raise ValueError('Dataset manifest rows differ from source audits') from exc
    return actual, meta
