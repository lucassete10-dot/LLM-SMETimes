"""Official CC-BY raw cycling archive; immutable download with checksum audit."""
from datetime import datetime,timezone
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
from pathlib import Path
import shutil
import traceback
import zipfile
import requests
from scripts.prepare_same_type_forecast import save_json

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--file',default='Dataset_1_NCA_battery.zip')
    parser.add_argument('--ranged',action='store_true')
    args=parser.parse_args()
    out=Path('data/raw/voltage_relaxation')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True);shutil.copy2(__file__,out/Path(__file__).name)
    try:
        response=requests.get('https://zenodo.org/api/records/6379165',timeout=30)
        response.raise_for_status();record=response.json()
        save_json(out/'official_record.json',record)
        assert record['metadata']['license']['id']=='cc-by-4.0'
        item=next(f for f in record['files'] if f['key']==args.file)
        assert Path(item['key']).name==item['key']
        dest=out/item['key'];md5=hashlib.md5();sha=hashlib.sha256();size=0
        save_json(out/'intake_manifest.json',dict(complete=False,file=item,record_id=record['id'],training_eligible=False))
        if args.ranged:
            parts=out/'parts';parts.mkdir();width=1024*1024
            url=f'https://zenodo.org/records/{record["id"]}/files/{item["key"]}?download=1'
            def download_part(start):
                end=min(start+width,item['size'])-1
                for attempt in range(3):
                    try:
                        r=requests.get(url,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(30,45))
                        try:
                            r.raise_for_status()
                            assert r.status_code==206 and r.headers.get('Content-Range')==f'bytes {start}-{end}/{item["size"]}'
                            content=r.content;assert len(content)==end-start+1
                        finally:r.close()
                        with (parts/f'{start:012d}.part').open('xb') as h:h.write(content)
                        return len(content)
                    except (requests.RequestException,AssertionError):
                        if attempt==2:raise
            completed=0
            with ThreadPoolExecutor(max_workers=4) as pool:
                jobs=[pool.submit(download_part,start) for start in range(0,item['size'],width)]
                for job in as_completed(jobs):
                    completed+=job.result()
                    if completed//(32*width)>(completed-width)//(32*width):print('Verified range MiB',completed//width,flush=True)
            with dest.open('xb') as handle:
                for part in sorted(parts.iterdir()):
                    content=part.read_bytes();handle.write(content);md5.update(content);sha.update(content);size+=len(content)
        else:
            r=requests.get(item['links']['self'],stream=True,timeout=(30,90));r.raise_for_status()
            with dest.open('xb') as handle:
                for chunk in r.iter_content(1024*1024):
                    if not chunk:continue
                    handle.write(chunk);md5.update(chunk);sha.update(chunk);size+=len(chunk)
                    if size%(64*1024*1024)<len(chunk):print('Downloaded MiB',size//1024//1024,flush=True)
        assert size==item['size'],(size,item['size'])
        assert 'md5:'+md5.hexdigest()==item['checksum']
        with zipfile.ZipFile(dest) as archive:
            members=[dict(name=i.filename,bytes=i.file_size,compressed=i.compress_size) for i in archive.infolist()]
            readmes={i.filename:archive.read(i).decode('utf-8',errors='replace') for i in archive.infolist()
                if not i.is_dir() and i.file_size<200000 and any(t in i.filename.lower() for t in ('readme','.txt'))}
        save_json(out/'archive_inventory.json',dict(members=members,readmes=readmes))
        save_json(out/'intake_manifest.json',dict(complete=True,training_eligible=False,
            reason='download verified; cell identity, cycle semantics and first100 input coverage await audit',
            file=item['key'],bytes=size,sha256=sha.hexdigest(),md5=md5.hexdigest(),
            official_checksum=item['checksum'],record_id=record['id'],doi=record['doi'],license='CC BY 4.0',
            source_url=item['links']['self']))
        print('OUTPUT='+str(out.resolve()),'Verified archive members='+str(len(members)),flush=True)
    except BaseException:
        save_json(out/'failure.json',dict(traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
