"""Content fingerprints for generated analysis tables and their inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def goertzel_code_fingerprint():
    root = Path(__file__).resolve().parent.parent
    paths = [root / 'plot_goertzel_vs_raw.py', *sorted((root / 'lilia').glob('*.py'))]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_goertzel_metadata(table, source, parameters):
    metadata = {
        'schema_version': 1,
        'source_path': str(Path(source).resolve()),
        'source_sha256': file_sha256(source),
        'table_sha256': file_sha256(table),
        'code_sha256': goertzel_code_fingerprint(),
        'parameters': parameters,
    }
    Path(str(table) + '.meta.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def validate_goertzel_cache(table, source, expected_parameters):
    """Reject stale/legacy tables; callers may check a subset of parameters."""
    sidecar = Path(str(table) + '.meta.json')
    if not sidecar.exists():
        raise ValueError('Cache metadata missing; recompute without --reuse-csv')
    metadata = json.loads(sidecar.read_text(encoding='utf-8'))
    valid = (metadata.get('schema_version') == 1
             and metadata.get('source_sha256') == file_sha256(source)
             and metadata.get('table_sha256') == file_sha256(table)
             and metadata.get('code_sha256') == goertzel_code_fingerprint()
             and all(metadata.get('parameters', {}).get(k) == v
                     for k, v in expected_parameters.items()))
    if not valid:
        raise ValueError('Cache source, settings, code or table changed; recompute without --reuse-csv')
