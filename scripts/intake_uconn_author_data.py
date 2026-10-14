"""Pin and fetch the original authors' 64-cell LFP processed data."""
import hashlib
import json
from pathlib import Path
import time
import requests
from scripts.prepare_same_type_forecast import save_json


def main():
    root=Path('data/raw/five_dataset_public/UConn_author');root.mkdir(parents=True,exist_ok=True)
    api='https://api.github.com/repos/REIL-UConn/fine-tuning-for-rapid-soh-estimation/git/trees/main?recursive=1'
    session=requests.Session();treefile=root/'repository_tree.json'
    if treefile.exists():
        tree=json.loads(treefile.read_text())
    else:
        r=session.get(api,timeout=30);r.raise_for_status();tree=r.json();save_json(treefile,tree)
    commit=tree['sha']
    paths=['README.md','LICENSE','scripts/config.py','scripts/data_processing/common_methods.py',
        'scripts/data_processing/postprocessing.py','processed_data/UConn-ILCC-LFP/data_slowpulse_1.pkl',
        'processed_data/UConn-ILCC-LFP/data_cc_1.pkl']
    entries={f['path']:f for f in tree['tree']};items=[]
    for name in paths:
        entry=entries[name];path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            partial=path.with_suffix(path.suffix+'.part')
            url=f'https://raw.githubusercontent.com/REIL-UConn/fine-tuning-for-rapid-soh-estimation/{commit}/{name}'
            offset=partial.stat().st_size if partial.exists() else 0
            if offset>entry['size']:raise ValueError('partial exceeds publisher length')
            headers={'Range':f'bytes={offset}-'} if offset else {}
            with session.get(url,headers=headers,stream=True,timeout=(25,180)) as response:
                response.raise_for_status();last=time.monotonic()
                append=offset>0 and response.status_code==206
                if append and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                    raise ValueError('range response mismatch')
                with partial.open('ab' if append else 'wb') as f:
                    for chunk in response.iter_content(128*1024):
                        if chunk:f.write(chunk)
                        if time.monotonic()-last>20:
                            print(f'DOWNLOAD {name} {f.tell()}',flush=True);last=time.monotonic()
            assert partial.stat().st_size==entry['size']
            content=partial.read_bytes();h=hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()
            assert h==entry['sha'];partial.rename(path)
        content=path.read_bytes();h=hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest();assert h==entry['sha']
        items.append(dict(path=name,size=len(content),git_blob=h,sha256=hashlib.sha256(content).hexdigest()))
        save_json(root/'manifest.json',dict(source=api,tree=commit,items=items,complete=False,
            dataset_identity='UConn-ISU-ILCC LFP 64 cells, confirmed by author scripts/config.py',
            note='Processed pulse data; source-cycle availability and preprocessing causality not yet audited.'))
        print('VERIFIED '+name,flush=True)
    save_json(root/'manifest.json',dict(source=api,tree=commit,items=items,complete=True,
        dataset_identity='UConn-ISU-ILCC LFP 64 cells',raw_cycling_downloaded=False,
        training_ready=False,need_audit='cycle coordinates, preprocessing, pulse-to-capacity alignment, cell coverage'))
    print('COMPLETE '+str(root.resolve()),flush=True)


if __name__=='__main__':main()
