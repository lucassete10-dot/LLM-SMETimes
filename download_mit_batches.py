"""Resumable verified downloads of original MIT Batch 2/3 MAT files.

The URLs are the canonical data.matr.io file API, not a processed-data mirror.
Parts are retained for auditable resumption; this script never deletes data.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path

import h5py
import requests


ROOT = Path(r'D:\LLM\MIT\raw')
FILES = {
    2: ('2017-06-30_batchdata_updated_struct_errorcorrect.mat', '5c86bf13fa2ede00015ddd82'),
    3: ('2018-04-12_batchdata_updated_struct_errorcorrect.mat', '5c86bd64fa2ede00015ddbb2'),
}
CHUNK = 32 * 1024 * 1024


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def download(batch, workers):
    name, file_id = FILES[batch]
    url = f'https://data.matr.io/1/api/v1/file/{file_id}/download'
    root = ROOT / f'batch{batch}_parts'
    ROOT.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=True)
    probe = requests.get(url, headers={'Range': 'bytes=0-0'}, timeout=30)
    probe.raise_for_status()
    if probe.status_code != 206:
        raise RuntimeError('Server did not honor byte ranges')
    total = int(probe.headers['Content-Range'].split('/')[-1])
    target = ROOT / name
    if target.exists() and target.stat().st_size == total:
        print('Existing complete file', target, flush=True)
    else:
        spans = [(index, start, min(total - 1, start + CHUNK - 1))
                 for index, start in enumerate(range(0, total, CHUNK))]

        def fetch(span):
            index, start, end = span
            part = root / f'{index:04d}.part'
            expected = end - start + 1
            if part.exists() and part.stat().st_size == expected:
                return index
            for attempt in range(5):
                try:
                    with requests.get(url, headers={'Range': f'bytes={start}-{end}'},
                                      stream=True, timeout=(30, 120)) as response:
                        response.raise_for_status()
                        content_range = response.headers.get('Content-Range', '')
                        if response.status_code != 206 or not content_range.startswith(f'bytes {start}-{end}/{total}'):
                            raise RuntimeError(f'Wrong range response: {content_range}')
                        temporary = part.with_suffix('.partial')
                        with temporary.open('wb') as output:
                            for block in response.iter_content(2**20):
                                output.write(block)
                        if temporary.stat().st_size != expected:
                            raise RuntimeError('Incomplete part')
                        temporary.replace(part)
                    print(f'Batch {batch} part {index+1}/{len(spans)}', flush=True)
                    return index
                except (requests.RequestException, RuntimeError) as error:
                    print(f'Batch {batch} part {index+1} retry {attempt+1}: {error}', flush=True)
                    if attempt == 4:
                        raise

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(fetch, spans))
        temporary = target.with_suffix('.mat.assembled')
        with temporary.open('wb') as output:
            for index, _, _ in spans:
                with (root / f'{index:04d}.part').open('rb') as source:
                    for block in iter(lambda: source.read(2**20), b''):
                        output.write(block)
        if temporary.stat().st_size != total:
            raise RuntimeError('Assembled size mismatch')
        temporary.replace(target)
    with h5py.File(target, 'r') as mat:
        cells = len(mat['batch']['summary'])
    manifest = {'batch': batch, 'file': str(target), 'source': url,
                'bytes': total, 'sha256': digest(target), 'cells': cells}
    (ROOT / f'batch{batch}_manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Verified', json.dumps(manifest), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('batch', type=int, choices=FILES)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    download(args.batch, args.workers)
