"""Complete four-metric and coverage reporting from all saved NASA curves."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text());rows=[]
    for path in sorted(root.rglob('*_predictions.npz')):
        if 'source_snapshot' in path.parts:continue
        role=path.name.removesuffix('_predictions.npz');fold=next(p['name'] for p in cfg['plans'] if p['name'] in path.parts);digest=sha(path);candidate=next((v for v in path.parts if v.startswith('candidate_')),None)
        with np.load(path) as a:
            keys=sorted({v.split('__100__')[0] for v in a.files});stored=json.loads(path.with_name(role+'_metrics.json').read_text());indexed={r['cell_id']:r for r in stored};assert set(indexed)==set(keys)
            for key in keys:
                c,y,p=[a[key+'__100__'+f] for f in ('cycle','truth','prediction')];assert np.all(np.diff(c)>0) and np.isfinite(y).all() and (y>0).all() and np.isfinite(p).all();e=p-y;ss=float(np.sum((y-y.mean())**2));mape=float(100*np.mean(abs(e)/y));rmse=float(np.sqrt(np.mean(e**2)))
                for f,v in [('mape_percent',mape),('rmse_soh',rmse)]:np.testing.assert_allclose(v,indexed[key][f],rtol=0,atol=1e-10)
                rows.append(dict(fold=fold,cell_id=key,role=role,candidate=candidate,seed=indexed[key].get('seed',cfg.get('seed')),mae_soh=float(np.mean(abs(e))),rmse_soh=rmse,mape_percent=mape,r2=1-float(np.sum(e**2))/ss if ss>0 else None,r2_undefined=ss==0,points=len(c),future_observation_span=float(c[-1]-c[0]),short_coverage=bool(len(c)<10 or c[-1]-c[0]<100),source_file=str(path.resolve()),source_sha256=digest))
    assert rows;save_json(root/'comprehensive_curve_metrics.json',dict(rows=rows,count=len(rows),role='reporting only, no model selection/exclusions',source_sha256=sha(__file__),goal_complete=False));print('SAVED_FULL_METRICS '+str(len(rows))+' '+str(root))


if __name__=='__main__':main()
