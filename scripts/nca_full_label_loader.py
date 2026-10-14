"""Add verified recovered early labels without changing historical caches."""
import json
from pathlib import Path
import numpy as np
from scripts.train_continuous_length_pilot import load_nominal
from scripts.prepare_same_type_forecast import sha


def load(plan,index_path):
    cells,conversion=load_nominal(plan)
    index_path=Path(index_path);index=json.loads(index_path.read_text());assert index['complete']
    for cid,cell in cells.items():
        if plan['dataset']!='Zhu2022_NCA':continue
        entry=index['cells'][cid];path=Path(entry['path']);assert sha(path)==entry['sha256']
        with np.load(path,allow_pickle=False) as data:
            c=data['cycle'].copy();y=data['soh_nominal'].copy()
        assert len(c)==len(y) and np.all(c<=100) and np.all(np.diff(c)>0)
        assert np.all(np.asarray(cell['cycle'])>100)
        cell['cycle']=np.r_[c,cell['cycle']];cell['truth']=np.r_[y,cell['truth']]
        assert np.all(np.diff(cell['cycle'])>0)
    return cells,conversion
