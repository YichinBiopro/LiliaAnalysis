"""
merge_subject_csvs.py
=====================
For each subject folder under iBrainCenter/ and YoGa/, merge all CSV files
into a single merged CSV sorted by absolute time.

Each CSV has the format:
    row 0: Device, ...
    row 1: Amp Gain, 500, Abs Time Offset[us], <offset_us>, ...
    row 2: Channels, 1, 2, 3, 4
    row 3: Sample Rate (per channel), ...
    row 4: Time[us], ch1, ch2, ch3, ch4
    row 5+: data (Time[us] is relative; absolute = Time[us] + Abs Time Offset[us])

The merged CSV:
  - Uses the same 4-row header as the earliest (lowest offset) source file,
    with Abs Time Offset[us] set to the minimum offset across all sources.
  - Data rows are sorted by absolute Time[us].

Usage
-----
    python merge_subject_csvs.py [--roots <dir> ...] [--outdir merged]

Defaults to scanning iBrainCenter/ and YoGa/ relative to the script location.
Merged files are written to <subject_folder>/merged.csv by default.
"""

import argparse
import os
import tempfile
import warnings

import pandas as pd

from lilia.io import read_abs_time_offset, read_lilia_frame
from lilia.pathing import get_project_root

BASE_DIR = get_project_root()


N_HEADER_ROWS = 4   # rows before the column-name row


def read_header_lines(path):
    """Return the first N_HEADER_ROWS raw lines of the file."""
    lines = []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= N_HEADER_ROWS:
                break
            lines.append(line.rstrip('\n'))
    return lines


def patch_offset_in_header(header_lines, new_offset):
    """
    Replace the Abs Time Offset[us] value in row 1 with *new_offset*.
    Also update Recording Start time[us] to match.
    """
    parts = header_lines[1].split(',')
    # col 3: Abs Time Offset value
    parts[3] = str(new_offset)
    # col 5: Recording Start time value (if present)
    if len(parts) > 5:
        parts[5] = str(new_offset)
    header_lines[1] = ','.join(parts)
    return header_lines


def load_csv_data(path, offset_us):
    """
    Load data rows (skip N_HEADER_ROWS + 1 column-name row).
    Returns DataFrame with absolute Time[us] in the first column.
    """
    df = read_lilia_frame(path, drop_invalid_time=True)
    col0 = df.columns[0]
    df[col0] = df[col0].astype('int64') + offset_us
    return df


def merge_subject(csv_paths, out_path):
    """Merge all CSVs for one subject and write to *out_path*."""
    csv_paths = [p for p in csv_paths if os.path.abspath(p) != os.path.abspath(out_path)]
    if not csv_paths:
        raise ValueError("No raw recordings to merge")
    # Read offsets
    offsets = {p: read_abs_time_offset(p) for p in csv_paths}
    min_offset = min(offsets.values())

    # Use header from the file with the earliest absolute start time
    earliest = min(offsets, key=offsets.get)
    header_lines = read_header_lines(earliest)
    header_lines = patch_offset_in_header(header_lines, min_offset)

    # Load and concatenate data, converting to absolute time
    frames = [load_csv_data(p, offsets[p]) for p in csv_paths]
    reference = read_header_lines(csv_paths[0])[2:4]
    for path, frame in zip(csv_paths, frames):
        if list(frame.columns) != list(frames[0].columns) or read_header_lines(path)[2:4] != reference:
            raise ValueError(f'Channel/sample-rate schema differs: {path}')
    merged = pd.concat(frames, ignore_index=True)

    # Sort by absolute Time[us] and drop only fully-identical rows. Dropping on
    # the timestamp alone would silently discard distinct samples from different
    # files that happen to collide on time.
    time_col = merged.columns[0]
    merged.sort_values(time_col, inplace=True, kind='stable')
    merged.drop_duplicates(inplace=True)
    merged.reset_index(drop=True, inplace=True)

    # Write: header lines first, then data (no extra index column)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    header_lines[0] += ',Time Basis,UTC'
    fd, temporary = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(out_path)), suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            for line in header_lines:
                f.write(line + '\n')
            merged.to_csv(f, index=False)
        os.replace(temporary, out_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    collisions = int(merged.iloc[:, 0].duplicated().sum())
    if collisions:
        warnings.warn(f'{collisions} timestamp collisions retained; resolve before signal analysis')

    duration_s = (merged.iloc[-1, 0] - merged.iloc[0, 0]) / 1e6
    print(f'  → {out_path}  '
          f'({len(csv_paths)} files, {len(merged)} rows, {duration_s:.1f} s)')


def find_subject_dirs(root):
    """Yield all immediate subdirectories of *root*."""
    for entry in sorted(os.scandir(root), key=lambda e: e.name):
        if entry.is_dir():
            yield entry.path


def collect_csvs(subject_dir, outname='merged.csv'):
    """Discover raw recording schemas, excluding outputs and analysis tables."""
    out_path = os.path.abspath(os.path.join(subject_dir, outname))
    sources = []
    for entry in sorted(os.scandir(subject_dir), key=lambda e: e.name):
        if not entry.is_file() or not entry.name.lower().endswith('.csv'):
            continue
        if entry.name.lower() == 'merged.csv' or os.path.abspath(entry.path) == out_path:
            continue
        with open(entry.path, encoding='utf-8-sig') as handle:
            lines = [handle.readline().strip() for _ in range(6)]
        if ('Time Basis,UTC' in lines[0] or ',Processing,' in lines[0]
                or not lines[4].lower().startswith('time[us],')):
            continue
        if 'Abs Time Offset[us]' not in lines[1]:
            continue
        offset = read_abs_time_offset(entry.path)
        # Legacy merged files predate the explicit Time Basis marker.
        try:
            first_time = float(lines[5].split(',')[0])
        except ValueError:
            first_time = 0
        if offset > 1e12 and first_time >= offset:
            warnings.warn(f'Skipping already-absolute recording: {entry.path}')
            continue
        sources.append(entry.path)
    return sources


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description='Merge per-subject EEG CSV files.')
    p.add_argument('--roots', nargs='+',
                   default=[os.path.join(script_dir, 'iBrainCenter'),
                            os.path.join(script_dir, 'YoGa')],
                   metavar='DIR',
                   help='Root directories to scan (default: iBrainCenter YoGa).')
    p.add_argument('--outname', default='merged.csv', metavar='NAME',
                   help='Output filename inside each subject folder (default: merged.csv).')
    return p.parse_args()


def main() -> None:
    """Discover all subject directories and merge their CSV files."""
    args = parse_args()
    for root in args.roots:
        if not os.path.isdir(root):
            print(f'Skipping (not found): {root}')
            continue
        print(f'\n[{os.path.basename(root)}]')
        for subj_dir in find_subject_dirs(root):
            csvs = collect_csvs(subj_dir, args.outname)
            if not csvs:
                continue
            subj_name = os.path.basename(subj_dir)
            out_path  = os.path.join(subj_dir, args.outname)
            if len(csvs) == 1:
                print(f'  {subj_name}: single file — skipping merge, '
                      f'copying reference only')
                # Still produce merged.csv for a uniform downstream interface
            print(f'  {subj_name}: merging {len(csvs)} file(s)...')
            merge_subject(csvs, out_path)


if __name__ == '__main__':
    main()
