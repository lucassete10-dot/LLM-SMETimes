"""Checksum-verified public mirror intake; does not prepare labels or fit models."""
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile
import requests
from scripts.prepare_same_type_forecast import save_json

ROOT = Path('data/raw/five_dataset_public')


def download(session, record, entry, folder):
    path = folder/entry['key']; partial = folder/(entry['key']+'.part')
    checksum = entry['checksum']; algorithm, expected = checksum.split(':')
    def digest(p):
        h=hashlib.new(algorithm)
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
        return h.hexdigest()
    if path.exists():
        assert path.stat().st_size==entry['size'] and digest(path)==expected
        return dict(file=str(path.resolve()),checksum=checksum,verified=True)
    offset=partial.stat().st_size if partial.exists() else 0
    assert shutil.disk_usage(folder).free>entry['size']-offset+512*1024*1024
    headers={'Range':f'bytes={offset}-'} if offset else {}
    started=time.monotonic();last=started
    with session.get(entry['links']['self'],stream=True,headers=headers,timeout=(25,60)) as r:
        r.raise_for_status()
        if offset and r.status_code!=206:
            raise ValueError('server did not honor resume; partial retained')
        with partial.open('ab' if offset else 'wb') as f:
            for chunk in r.iter_content(4*1024*1024):
                if chunk:f.write(chunk);offset+=len(chunk)
                if time.monotonic()-last>20:
                    print(f'DOWNLOAD {entry["key"]} {offset}/{entry["size"]}',flush=True)
                    save_json(folder/'progress.json',dict(file=entry['key'],bytes=offset,total=entry['size'],terminal=False))
                    last=time.monotonic()
    assert partial.stat().st_size==entry['size'], 'incomplete length'
    assert digest(partial)==expected, 'checksum mismatch; partial retained'
    partial.rename(path)
    with zipfile.ZipFile(path) as archive:
        members=[dict(name=i.filename,size=i.file_size) for i in archive.infolist()]
    save_json(folder/(entry['key']+'.inventory.json'),members)
    return dict(file=str(path.resolve()),checksum=checksum,verified=True,seconds=time.monotonic()-started,members=len(members))


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    session=requests.Session()
    session.headers['User-Agent']='Battery-SOH-research-intake/1.0'
    url='https://zenodo.org/api/records/21149533'
    response=session.get(url,timeout=30);response.raise_for_status();record=response.json()
    folder=ROOT/'BatteryLife_v12';folder.mkdir(exist_ok=True)
    save_json(folder/'source_record.json',record)
    files={f['key']:f for f in record['files']}
    manifest=dict(source=url,original_hust='https://data.mendeley.com/datasets/nsc7hnsg4s/2',
        status='in_progress',items=[],original_access_403=True,
        uconn_requested='https://digitalcommons.lib.uconn.edu/reil_datasets/1/',
        note='BatteryLife ISU_ILCC identity not yet verified against requested UConn dataset; not substituted.')
    save_json(folder/'manifest.json',manifest)
    try:
        for key in ('READMEs.zip','HUST.zip'):
            manifest['items'].append(download(session,record,files[key],folder))
            save_json(folder/'manifest.json',manifest)
        manifest['status']='hust_mirror_download_verified_not_prepared'
        save_json(folder/'manifest.json',manifest)
        save_json(folder/'progress.json',dict(terminal=True,success=True))
        print('COMPLETE '+str(folder.resolve()),flush=True)
    except Exception as e:
        manifest['status']='interrupted';manifest['error']=f'{type(e).__name__}: {e}'
        save_json(folder/'manifest.json',manifest)
        save_json(folder/'progress.json',dict(terminal=True,success=False,error=manifest['error']))
        raise


if __name__=='__main__':main()
