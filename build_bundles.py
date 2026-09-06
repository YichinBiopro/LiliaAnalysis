"""Generate distributable copies from canonical sources; --check detects drift."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def bundle_files(root: Path):
    bundle = root / 'plot_index_vs_raw_bundle'
    for source in sorted((root / 'lilia').glob('*.py')):
        yield source, bundle / 'lilia' / source.name
    for name in ('plot_index_vs_raw.py', 'plot_event_markers.py', 'merge_subject_csvs.py'):
        yield root / name, bundle / name
    yield root / 'lilia' / 'quality.py', root / 'signal_quality_package' / 'quality.py'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Report drift without writing files')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    changed = []
    for source, destination in bundle_files(root):
        if destination.exists() and source.read_bytes() == destination.read_bytes():
            continue
        changed.append(str(destination.relative_to(root)))
        if not args.check:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    if args.check and changed:
        parser.exit(1, 'Bundle drift:\n' + '\n'.join(changed) + '\n')
    print(f'{len(changed)} copies {"differ" if args.check else "updated"}')


if __name__ == '__main__':
    main()
