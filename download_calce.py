"""Download original CALCE CS2 archives linked by the UMD official data page."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import zipfile
import requests

ROOT = Path(r'D:\LLM\CALCE\raw')
IDS = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']


def download(cell):
    url = f'https://web.calce.umd.edu/batteries/data/{cell}.zip'
    target = ROOT / f'{cell}.zip'
    ROOT.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temp = target.with_suffix('.zip.partial')
        probe = requests.get(url, headers={'Range': 'bytes=0-0'}, timeout=30)
        probe.raise_for_status()
        total = int(probe.headers['Content-Range'].split('/')[-1])
        chunk_size = 2*1024*1024
        def part(start):
            end = min(total-1, start+chunk_size-1)
            for attempt in range(3):
                try:
                    r = requests.get(url, headers={'Range': f'bytes={start}-{end}'}, timeout=(15,30))
                    r.raise_for_status()
                    if r.status_code != 206 or len(r.content) != end-start+1:
                        raise ValueError('Range response mismatch')
                    return r.content
                except Exception:
                    if attempt == 2:
                        raise
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            with temp.open('wb') as handle:
                for block in pool.map(part, range(0,total,chunk_size)):
                    handle.write(block)
        temp.replace(target)
    with zipfile.ZipFile(target) as archive:
        error = archive.testzip()
        if error:
            raise ValueError(error)
        dest = (ROOT/'extracted'/cell).resolve()
        for info in archive.infolist():
            final = (dest/info.filename).resolve()
            if not final.is_relative_to(dest):
                raise ValueError('Unsafe zip path')
        archive.extractall(dest)
        members = archive.namelist()
    result = {'cell': cell, 'url': url, 'size_bytes': target.stat().st_size,
              'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'members': members}
    print(cell, result['size_bytes'], len(members), members[:4], flush=True)
    return result


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(download, IDS))
    (ROOT/'download_manifest.json').write_text(json.dumps(results, indent=2))
