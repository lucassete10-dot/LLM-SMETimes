"""Locked TRAIN LOO routes, one tangent-extension comparison, zero fitting."""
import json,shutil
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from battery_smetimes.mit_template_tangent import curve
from scripts.diagnose_mit_template_route_loo import evaluate
from scripts.prepare_same_type_forecast import sha,save_json

PARENT=Path('experiments/mit_template_route_loo/20261007T054746_835336Z')


def main():
    cfg=json.loads((PARENT/'config.json').read_text());assert json.loads((PARENT/'independent_verification.json').read_text())['passed']
    cache=joblib.load(cfg['cache_file']);templates={t['cell_id']:t for t in cache['templates']}
    routes=json.loads((PARENT/'rows.json').read_text());ids=cfg['plan']['train']
    root=Path('experiments/mit_template_tangent_diagnostic')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True);snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),Path('battery_smetimes/mit_template_tangent.py'),Path('docs/soh_mit_template_tangent_diagnostic_20261007.md')]
    for f in files:shutil.copy2(f,snap/f.name)
    deps=[PARENT/'config.json',PARENT/'independent_verification.json',PARENT/'rows.json',PARENT/'curves.npz',Path(cfg['cache_file'])]
    save_json(root/'config.json',dict(parent=str(PARENT.resolve()),input_sha256={str(f.resolve()):sha(f) for f in deps},plan=cfg['plan'],
        source_sha256={str(f):sha(f) for f in files},new_fits=0,validation_loaded=False,outer_loaded=False,adopted=False))
    rows=[];arrays={}
    with np.load(PARENT/'curves.npz',allow_pickle=False) as old:
        for held,rr in zip(ids,routes):
            assert rr['cell_id']==held and held not in rr['neighbor_ids']
            c=old[held+'__cycle'];truth=old[held+'__truth_change'];baseline=old[held+'__nearest_change']
            changes=[]
            for k in rr['neighbor_ids']:
                t=templates[k];changes.append(curve(t['parameters'],t['record_end'],c)-curve(t['parameters'],t['record_end'],[101.])[0])
            p=np.sum(np.stack(changes)*np.asarray(rr['weights'])[:,None],axis=0)
            rows.append(dict(cell_id=held,baseline=evaluate(baseline,truth),tangent=evaluate(p,truth),
                majority_neighbors_extrapolated=rr['majority_neighbors_extrapolated']))
            arrays.update({held+'__'+f:v for f,v in [('cycle',c),('truth_change',truth),('baseline_change',baseline),('tangent_change',p)]})
    fields=tuple(rows[0]['baseline']);summary=dict(cells=len(rows),curves=166,new_fits=0,
        **{arm:{f:float(np.mean([r[arm][f] for r in rows])) for f in fields} for arm in ('baseline','tangent')},
        endpoint_improved_cells=sum(r['tangent']['change_endpoint_pp']<r['baseline']['change_endpoint_pp'] for r in rows),
        validation_loaded=False,outer_loaded=False,adopted=False,goal_complete=False,
        interpretation='TRAIN-held relative-change diagnostic only; same prelocked routes, donor end controls extrapolation not routing. No absolute SOH accuracy claim.')
    save_json(root/'rows.json',rows);np.savez_compressed(root/'curves.npz',**arrays);save_json(root/'summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
