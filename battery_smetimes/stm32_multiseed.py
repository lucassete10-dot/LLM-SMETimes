"""Same compact numeric contract, extended to the isolated NASA cohort."""
import numpy as np
import torch
from battery_smetimes.forecast import query_matrix
from battery_smetimes.research_baselines import signal_prefix
from battery_smetimes.stm32_forecast import RAW_COLUMNS as ORIGINAL_COLUMNS, predict_int8

RAW_COLUMNS={**ORIGINAL_COLUMNS,'NASA':[0,5,10,11,16,22,32]}


def compact_input(cell,origin,future):
    width=cell['raw'].shape[1];prefix=signal_prefix(cell,origin)
    selected=[b*width+c for b in range(4) for c in RAW_COLUMNS[cell['metadata']['dataset']]]
    selected.extend(range(4*width,len(prefix)+3))
    result=query_matrix(prefix,origin,future)[:,selected]
    if result.shape[1]!=40:raise ValueError('compact feature schema mismatch')
    return result


def pairs(cells,ids):
    xs=[];ys=[];ws=[]
    for key in ids:
        c=cells[key];idx=np.flatnonzero(np.isfinite(c['soh'])&(c['soh']>0)&(c['label_cycle']>100))
        if len(idx)<2:raise ValueError('insufficient training future targets')
        idx=idx[np.unique(np.linspace(0,len(idx)-1,min(128,len(idx))).round().astype(int))]
        xs.append(compact_input(c,100,c['label_cycle'][idx]));ys.append(c['soh'][idx]);ws.append(np.full(len(idx),1/len(ids)/len(idx)))
    return np.concatenate(xs),np.concatenate(ys),np.concatenate(ws)


def predict(bundle,cell,origin,future):
    if origin!=100 or cell['physical_family']!=bundle['physical_family']:raise ValueError('origin or physical family mismatch')
    x=bundle['scale'].transform(compact_input(cell,origin,future))
    if bundle.get('quantized'):return predict_int8(bundle['int8_layers'],x)
    with torch.inference_mode():return bundle['network'](torch.from_numpy(x)).numpy()
