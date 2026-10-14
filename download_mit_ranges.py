"""Download Batch 1 ZIP with independent resumable HTTP range parts."""
import concurrent.futures
import json
from pathlib import Path
import requests

from download_mit import ROOT,NAME,URL,main as verify

PARTS=ROOT/'range_parts'
SIZE=16*1024*1024


def fetch(spec):
    index,start,end=spec
    file=PARTS/f'{index:04d}.part'
    expected=end-start+1
    if file.exists() and file.stat().st_size==expected:return index
    for attempt in range(5):
        try:
            with requests.get(URL,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(30,90)) as r:
                r.raise_for_status()
                if r.status_code!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
                    raise ValueError('Incorrect range response')
                temp=file.with_suffix('.partial')
                with temp.open('wb') as handle:
                    for chunk in r.iter_content(2**20):handle.write(chunk)
                if temp.stat().st_size!=expected:raise ValueError('Incomplete chunk')
                temp.replace(file)
            print('Part',index,'done',flush=True)
            return index
        except Exception as exc:
            print('Part',index,'retry',attempt+1,type(exc).__name__,flush=True)
            if attempt==4:raise


if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True)
    PARTS.mkdir(exist_ok=True)
    probe=requests.get(URL,headers={'Range':'bytes=0-0'},timeout=30)
    probe.raise_for_status()
    total=int(probe.headers['Content-Range'].split('/')[-1])
    specs=[(i,start,min(total-1,start+SIZE-1)) for i,start in enumerate(range(0,total,SIZE))]
    print('Total',total,'parts',len(specs),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(fetch,specs))
    target=ROOT/(NAME+'.zip')
    temp=target.with_suffix('.zip.assembled')
    with temp.open('wb') as out:
        for i,_,_ in specs:
            with (PARTS/f'{i:04d}.part').open('rb') as src:
                for block in iter(lambda:src.read(2**20),b''):out.write(block)
    if temp.stat().st_size!=total:raise RuntimeError('Assembly size mismatch')
    temp.replace(target)
    verify()
