"""Frozen TRAIN support versus already-viewed screenshot failures; no selection."""
import json
from pathlib import Path
import numpy as np
from scripts.audit_llm_curve_repair import load_frozen,training_inventory,fingerprint
from scripts.prepare_same_type_forecast import save_json


def main():
    root=Path(json.loads(Path('experiments/llm_curve_repair/latest_P0.json').read_text())['folder'])
    cfg=json.loads((root/'config.json').read_text());_,cells,plan,_,_=load_frozen(cfg)
    rows=training_inventory(cells,plan['train'],cfg['origin'])
    starts=[r['exact_origin_soh'] for r in rows if r['origin_label_available']]
    ends=[cells[k]['label_cycle'][-1] for k in plan['train']]
    features=np.stack([cells[k]['raw'][cells[k]['cycle']<=cfg['origin']][-1] for k in plan['train']])
    median=np.nanmedian(features,0);mad=np.nanmedian(abs(features-median),0);scale=np.maximum(mad,1e-5)
    records=[]
    for cid in ['b1c2','b1c18','b2c1','b2c2']:
        cell=cells[cid];at=np.flatnonzero(cell['label_cycle']==cfg['origin']);level=float(cell['soh'][at[0]]) if len(at) else None
        z=abs(cell['raw'][cell['cycle']<=cfg['origin']][-1]-median)/scale
        columns=np.argsort(np.nan_to_num(z,nan=-1))[-5:][::-1]
        records.append({'cell':cid,'scope':'already-viewed development screenshot, diagnosis only; no model selection',
            'origin_soh':level,'origin_soh_outside_TRAIN_range':bool(level<min(starts) or level>max(starts)) if level is not None else None,
            'last_source_cycle':float(cell['label_cycle'][-1]),
            'last_cycle_outside_TRAIN_range':bool(cell['label_cycle'][-1]<min(ends) or cell['label_cycle'][-1]>max(ends)),
            'largest_raw_prefix_last_observation_MAD_ratios':[{'column':int(i),'ratio':float(z[i])} for i in columns if np.isfinite(z[i])]})
    report={'TRAIN_cells':len(rows),'TRAIN_origin_soh_range':[min(starts),max(starts)],
        'TRAIN_last_cycle_range':[float(min(ends)),float(max(ends))],
        'TRAIN_categories':{c:sum(r.get('category')==c for r in rows) for c in ['fast','slow','accelerating','recovery']},
        'reference':'Initial comparable measured discharge, not nominal capacity',
        'screenshots':records,'raw_MAD_floor':1e-5,'source_sha256':fingerprint(Path(__file__)),
        'warning':'Large feature deviations suggest covariate shift but do not prove causal signal insufficiency. Target metadata used only to diagnose already-viewed failures, never inference or new split selection.'}
    save_json(root/'TRAIN_and_viewed_failures_support_audit.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
