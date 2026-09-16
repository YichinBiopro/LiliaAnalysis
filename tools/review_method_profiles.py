"""Render explicit Goertzel exclusions from existing method-freeze arrays."""
import argparse
import json
import os
from pathlib import Path

from freeze_method_profiles import sha, verify_files, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    capture, out = args.capture.resolve(), args.out.resolve()
    verify_files(json.loads((capture/'manifest_local.json').read_text())['files'], capture)
    out.mkdir(parents=True, exist_ok=False)
    (out/'.gitignore').write_text('local/\nmpl/\n!*.png\n')
    (out/'local').mkdir()
    os.environ['MPLCONFIGDIR'] = str(out/'mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    records, inputs = [], []
    for name in ('real_1', 'synthetic_gap', 'synthetic_short'):
        paths = [capture/'local'/label/(name+'.npz') for label in ('expected', 'actual')]
        inputs.extend(dict(path=str(p), sha256=sha(p)) for p in paths)
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
        counts = {}
        with np.load(paths[0]) as old, np.load(paths[1]) as new:
            for win, ax in zip(('5.0', '0.5'), axes):
                prefix = 'goertzel_' + win + '_'
                time, keep = new[prefix+'time_s'], new[prefix+'keep']
                groups = new[prefix+'segment_ids']
                counts[win] = dict(windows=len(keep), accepted=int(keep.sum()))
                ax.plot([], [], color='gray', marker='x', ls='', label='excluded current windows')
                ax.plot([], [], color='blue', label='frozen unmasked power')
                ax.plot([], [], color='orange', ls='--', label='current unmasked power')
                ax.plot([], [], color='green', marker='o', label='accepted legacy median 5')
                for sid in np.unique(groups):
                    rows = groups == sid
                    ax.plot(time[rows], old[prefix+'goertzel_db'][rows], color='blue', lw=1)
                    ax.plot(time[rows], new[prefix+'goertzel_db'][rows], '--', color='orange', lw=1)
                    ax.plot(time[rows], new[prefix+'smooth'][rows], '-o', color='green', ms=3)
                ax.scatter(time[~keep], new[prefix+'goertzel_db'][~keep], color='gray', marker='x', s=15)
                if not len(keep):
                    ax.text(.03, .7, 'No complete window', transform=ax.transAxes)
                elif not keep.any():
                    ax.text(.03, .9, 'All windows excluded by legacy policy', transform=ax.transAxes)
                ax.set(title=f'{win}s windows: accepted {int(keep.sum())}/{len(keep)}',
                       xlabel='Elapsed seconds', ylabel='Unnormalized Goertzel power (dB)')
                ax.legend(fontsize=8, loc='lower right')
        fig.suptitle(name + ': frozen/current power and unchanged quality selection')
        path = (out/'local' if name.startswith('real_') else out)/(name+'.png')
        fig.savefig(path, dpi=120)
        plt.close(fig)
        records.append(dict(case=name, path=str(path.relative_to(out)), sha256=sha(path), counts=counts))
    write_json(out/'plots.json', records)
    files = inputs + [dict(path=r['path'], sha256=r['sha256']) for r in records]
    files += [dict(path='plots.json', sha256=sha(out/'plots.json')),
              dict(path=str(Path(__file__).resolve()), sha256=sha(__file__))]
    write_json(out/'manifest_local.json', dict(schema_version=1, files=files))
    write_json(out/'manifest_repository.json', dict(schema_version=1,
        files=[r for r in files if not r['path'].startswith('local/') and r not in inputs]))


if __name__ == '__main__':
    main()
