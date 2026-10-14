"""Group diagnostics only; no new scoring weights or model selection."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.prepare_same_type_forecast import save_json,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);root=ap.parse_args().run
    assert json.loads((root/'independent_verification.json').read_text())['passed'];cfg=json.loads((root/'config.json').read_text());rows=json.loads((root/'comprehensive_curve_metrics.json').read_text())['rows'];rr=[r for r in rows if r['role']=='validation']
    fields=('mape_percent','late_mape','endpoint_pp','drop_error_pp','start_error_pp');by_candidate=[]
    for i,p in enumerate(cfg['candidates']):
        records=[r for r in rr if r['candidate']==i];assert len(records)==12
        by_candidate.append(dict(candidate=i,parameters=p,overlapping_validation_records=12,**{f:float(np.mean([r[f] for r in records])) for f in fields}))
    by_block=[]
    for block in ('V','VI','VIT'):
        candidates=[r for r in by_candidate[1:] if r['parameters']['block']==block]
        by_block.append(dict(block=block,**{f:dict(min=float(min(r[f] for r in candidates)),max=float(max(r[f] for r in candidates))) for f in fields}))
    # Compare like-for-like six-feature direct calibrators as well. Comparing
    # only against the old fused curve would confound features with architecture.
    parent=Path(cfg['parent']);grand=Path(json.loads((parent/'config.json').read_text())['parent']);paired=[]
    for i,p in enumerate(cfg['candidates'][1:],1):
        legacy=(3 if p['alpha']==10 else 5)+(6 if p['mode']=='delta' else 0)
        new=[r for r in rr if r['candidate']==i];old=[]
        for r in new:
            folder=grand/r['fold']/('hold_'+r['heldout'])/f'candidate_{legacy}'
            metric=json.loads((folder/'validation_metrics.json').read_text())[0];assert metric['cell_id']==r['cell_id']
            with np.load(folder/'validation_predictions.npz') as a:
                prefix=r['cell_id']+'__100__';metric['start_error_pp']=float(100*abs(a[prefix+'prediction'][0]-a[prefix+'truth'][0]))
            old.append(metric)
        paired.append(dict(candidate=i,parameters=p,legacy_six_feature_direct_candidate=legacy,
                           new={f:float(np.mean([r[f] for r in new])) for f in fields},
                           legacy={f:float(np.mean([r[f] for r in old])) for f in fields}))
    save_json(root/'feature_group_diagnostics.json',dict(candidates=by_candidate,blocks=by_block,matched_direct_comparisons=paired,independent_validation_cells=4,overlapping_records_per_candidate=12,not_outer_results=True,diagnostics_only=True,new_selection=False,source_sha256=sha(__file__)))
    print(json.dumps(dict(control=by_candidate[0],blocks=by_block,matched_feature_mape_improvements=sum(r['new']['mape_percent']<r['legacy']['mape_percent'] for r in paired),matched_feature_start_improvements=sum(r['new']['start_error_pp']<r['legacy']['start_error_pp'] for r in paired))),flush=True)

if __name__=='__main__':main()
