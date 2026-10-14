"""Download and verify the authors' original XJTU data archive from Zenodo."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import zipfile

import requests


ROOT = Path(r'D:\LLM\XJTU\raw')
URL = 'https://zenodo.org/records/10963339/files/Battery%20Dataset.zip?download=1'
EXPECTED_BYTES = 2438769934
EXPECTED_MD5 = 'a635cef9678e0de21f9d5c1f78a4342c'
CHUNK = 32 * 1024 * 1024


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    parts = ROOT / 'range_parts'
    parts.mkdir(exist_ok=True)
    target = ROOT / 'Battery Dataset.zip'
    probe = requests.get(URL, headers={'Range': 'bytes=0-0'}, timeout=30)
    probe.raise_for_status()
    total = int(probe.headers['Content-Range'].split('/')[-1])
    if probe.status_code != 206 or total != EXPECTED_BYTES:
        raise RuntimeError('Unexpected source size or range behavior')
    if not target.exists():
        spans = [(i, start, min(total - 1, start + CHUNK - 1))
                 for i, start in enumerate(range(0, total, CHUNK))]

        def fetch(span):
            index, start, end = span
            part = parts / f'{index:04d}.part'
            expected = end - start + 1
            if part.exists() and part.stat().st_size == expected:
                return index
            temporary = part.with_suffix('.partial')
            if temporary.exists() and temporary.stat().st_size > expected:
                raise RuntimeError(f'Oversized partial range: {temporary}')
            for attempt in range(20):
                try:
                    have = temporary.stat().st_size if temporary.exists() else 0
                    if have == expected:
                        temporary.replace(part)
                        return index
                    next_start = start + have
                    with requests.get(URL, headers={'Range': f'bytes={next_start}-{end}'},
                                      stream=True, timeout=(30, 120)) as response:
                        response.raise_for_status()
                        if response.status_code != 206 or not response.headers.get(
                                'Content-Range', '').startswith(f'bytes {next_start}-{end}/{total}'):
                            raise RuntimeError('Incorrect range response')
                        with temporary.open('ab') as output:
                            for block in response.iter_content(2**20):
                                output.write(block)
                        if temporary.stat().st_size != expected:
                            raise RuntimeError('Incomplete range')
                        temporary.replace(part)
                    print(f'Part {index+1}/{len(spans)} complete', flush=True)
                    return index
                except (requests.RequestException, RuntimeError) as error:
                    print(f'Part {index+1} retry {attempt+1}: {error}', flush=True)
                    if attempt == 19:
                        raise

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(fetch, spans))
        assembled = target.with_suffix('.zip.assembled')
        with assembled.open('wb') as output:
            for index, _, _ in spans:
                with (parts / f'{index:04d}.part').open('rb') as source:
                    for block in iter(lambda: source.read(2**20), b''):
                        output.write(block)
        if assembled.stat().st_size != total:
            raise RuntimeError('Assembled length mismatch')
        assembled.replace(target)
    digest = hashlib.md5()
    with target.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    if digest.hexdigest() != EXPECTED_MD5:
        raise RuntimeError('Official Zenodo MD5 checksum mismatch')
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Archive CRC failure')
        entries = [{'name': member.filename, 'bytes': member.file_size}
                   for member in archive.infolist() if not member.is_dir()]
    manifest = {'source': URL, 'zenodo_record': 'https://zenodo.org/records/10963339',
                'file': str(target), 'bytes': total, 'md5': digest.hexdigest(),
                'entries': entries}
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Verified', len(entries), 'archive entries', flush=True)
    print('Largest', sorted(entries, key=lambda row: row['bytes'], reverse=True)[:8], flush=True)


if __name__ == '__main__':
    main()
