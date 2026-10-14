"""Retrieve a pinned, public author notebook without executing its code."""
import base64
import hashlib
import json
from pathlib import Path
import requests
from scripts.prepare_same_type_forecast import save_json


def main():
    root=Path('data/raw/five_dataset_public/UConn_author')
    tree=json.loads((root/'repository_tree.json').read_text())
    record=next(r for r in tree['tree'] if r['path']=='notebooks/data_overview.ipynb')
    path=root/record['path']; url='https://api.github.com/repos/REIL-UConn/fine-tuning-for-rapid-soh-estimation/git/blobs/'+record['sha']
    if not path.exists():
        response=requests.get(url,timeout=(10,30)); response.raise_for_status()
        blob=response.json(); assert blob['encoding']=='base64'
        content=base64.b64decode(blob['content'])
        actual=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        assert actual==record['sha'] and len(content)==record['size']
        path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(content)
        save_json(root/'overview_provenance.json',dict(url=url,git_blob=actual,bytes=len(content),
                  sha256=hashlib.sha256(content).hexdigest(),notebook_executed=False))
    notebook=json.loads(path.read_text(encoding='utf-8'))
    for i,cell in enumerate(notebook['cells']):
        source=''.join(cell.get('source',[]))
        if any(key in source.lower() for key in ('num_cycles','num cycles','cumul','aging traject','cycle number')):
            print(f'CELL {i} {cell["cell_type"]}\n{source[:6000]}\n')


if __name__=='__main__': main()
