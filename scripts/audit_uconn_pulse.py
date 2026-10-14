"""Inspect pinned author pulse data using a minimal NumPy-only unpickler."""
import hashlib
import importlib
import json
import pickle
from pathlib import Path
import numpy as np
from scripts.prepare_same_type_forecast import save_json


class NumpyUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('numpy','dtype'),('numpy','ndarray'),('numpy.core.numeric','_frombuffer'),
                 ('numpy._core.numeric','_frombuffer'),('numpy.core.multiarray','_reconstruct'),
                 ('numpy._core.multiarray','_reconstruct'),('numpy.core.multiarray','scalar')}
        if (module,name) not in allowed:raise pickle.UnpicklingError('Disallowed global: '+module+'.'+name)
        return getattr(importlib.import_module(module),name)


def load():
    root=Path('data/raw/five_dataset_public/UConn_author');path=root/'processed_data/UConn-ILCC-LFP/data_slowpulse_1.pkl'
    tree=json.loads((root/'repository_tree.json').read_text());entry=next(r for r in tree['tree'] if r['path']=='processed_data/UConn-ILCC-LFP/data_slowpulse_1.pkl')
    raw=path.read_bytes();assert len(raw)==entry['size'];assert hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()==entry['sha']
    with path.open('rb') as f:data=NumpyUnpickler(f).load()
    assert isinstance(data,dict) and all(isinstance(v,np.ndarray) for v in data.values())
    return data


def main():
    data=load();report=dict(schema={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in data.items()},cells=[])
    for key in sorted(np.unique(data['cell_id'])):
        take=data['cell_id']==key;cycles=np.unique(data['num_cycles'][take]);rpt=np.unique(data['rpt'][take])
        report['cells'].append(dict(cell_id=int(key),rows=int(take.sum()),unique_cycles=cycles.tolist(),
            rpts=rpt.tolist(),groups=np.unique(data['group_id'][take]).tolist(),
            socs=np.unique(data['soc'][take]).tolist(),pulse_types=np.unique(data['pulse_type'][take]).tolist(),
            capacity_range=[float(np.min(data['q_dchg'][take])),float(np.max(data['q_dchg'][take]))]))
    report['num_cells']=len(report['cells']);report['rows']=len(data['cell_id'])
    report['nonfinite_counts']={k:int((~np.isfinite(v)).sum()) for k,v in data.items() if v.dtype.kind in 'fi'}
    def safe(v):
        if isinstance(v,dict):return {k:safe(x) for k,x in v.items()}
        if isinstance(v,list):return [safe(x) for x in v]
        if isinstance(v,float) and not np.isfinite(v):return None
        return v
    report=safe(report)
    save_json(Path('data/raw/five_dataset_public/UConn_author/pulse_schema_audit.json'),report)
    print(json.dumps({k:v for k,v in report.items() if k!='cells'},indent=2));print(json.dumps(report['cells'][:3],indent=2))


if __name__=='__main__':main()
