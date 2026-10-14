"""Frozen inner-model transformation screen; no outer loading or new fit."""
import json,logging,shutil
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.nasa_calibrated_level import predict
from battery_smetimes.nasa_cycle_calibration import predict as calibrate
from battery_smetimes.research_baselines import predict_baseline
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.audit_rate_blend_metrics import metrics
from scripts.verify_neighbor_curve_nca_formal import score
from scripts.prepare_same_type_forecast import sha,save_json


def main():
    parent=Path('experiments/nasa_cycle_calibration/20261005T154730_956753Z');assert json.loads((parent/'independent_verification.json').read_text())['passed'];pcfg=json.loads((parent/'config.json').read_text());root=Path('experiments/nasa_calibrated_level')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True);snapshot=root/'source_snapshot';snapshot.mkdir();files=[Path(__file__),Path('battery_smetimes/nasa_calibrated_level.py'),Path('docs/soh_nasa_calibrated_level_20261005.md')]
    for f in files:shutil.copy2(f,snapshot/f.name)
    save_json(root/'config.json',dict(parent=str(parent.resolve()),parent_config_sha256=sha(parent/'config.json'),parent_verification_sha256=sha(parent/'independent_verification.json'),source_sha256={str(f):sha(f) for f in files},plans=pcfg['plans'],seed=pcfg['seed'],original_test_loaded=False,goal_complete=False));logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(root/'screen.log',encoding='utf-8'),logging.StreamHandler()]);records=[]
    for plan in pcfg['plans']:
        cells=load_subset(plan['train']+plan['validation']);assert not set(cells)&set(plan['test']);folder=root/plan['name'];folder.mkdir();primary_file=parent/plan['name']/'candidate_0/model.joblib';primary=joblib.load(primary_file);results=[]
        for i in range(13):
            calibration_file=parent/plan['name']/f'candidate_{i}/model.joblib';calibration=joblib.load(calibration_file) if i else None;bundle=dict(primary=primary,calibration=calibration,train_ids=plan['train']);dest=folder/f'candidate_{i}';dest.mkdir();joblib.dump(bundle,dest/'model.joblib');save_json(dest/'provenance.json',dict(primary_file=str(primary_file.resolve()),primary_sha256=sha(primary_file),calibration_file=str(calibration_file.resolve()) if i else None,calibration_sha256=sha(calibration_file) if i else None));anchors=[]
            for role,keys in [('training',plan['train']),('validation',plan['validation'])]:
                arrays={};rows=[]
                for key in keys:
                    query=clean(cells[key]);c=cells[key]['label_cycle'];y=cells[key]['soh'];take=(c>100)&np.isfinite(y)&(y>0);c,y=c[take],y[take];p=predict(bundle,query,c);base=predict_baseline(primary,query,100,c);baseline101=float(predict_baseline(primary,query,100,[101.])[0]);cal101=float(calibrate(calibration,query,[101.])[0]) if i else baseline101;anchors.append(dict(cell_id=key,role=role,primary101=baseline101,calibrated101=cal101,offset_soh=cal101-baseline101));np.testing.assert_allclose(p[0]-p[-1],base[0]-base[-1],rtol=0,atol=1e-12)
                    rows.append(dict(cell_id=key,**metrics(c,y,p)));arrays.update({key+'__100__'+f:v for f,v in [('cycle',c),('truth',y),('prediction',p)]})
                np.savez_compressed(dest/(role+'_predictions.npz'),**arrays);save_json(dest/(role+'_metrics.json'),rows)
                if role=='validation':results.append(dict(candidate=i,score=score(rows),**{f:float(np.mean([r[f] for r in rows])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')}))
            save_json(dest/'anchors.json',anchors);logging.info('SAVED %s candidate%s',plan['name'],i)
        selected=min(results,key=lambda r:(r['score'],r['candidate']));record=dict(fold=plan['name'],selected=selected,control=results[0],candidates=results);records.append(record);save_json(folder/'selection.json',record)
    fields=('score','mape_percent','endpoint_pp','drop_error_pp');new={f:float(np.mean([r['selected'][f] for r in records])) for f in fields};old={f:float(np.mean([r['control'][f] for r in records])) for f in fields};wins=sum(r['selected']['score']<r['control']['score'] for r in records);expand=wins>=3 and new['endpoint_pp']<=old['endpoint_pp']+1e-10 and new['drop_error_pp']<=old['drop_error_pp']+1e-10 and new['mape_percent']<=old['mape_percent']+.1
    save_json(root/'summary.json',dict(records=records,new=new,control=old,wins=wins,expand=expand,outer_loaded=False,goal_complete=False));save_json(root/'progress.json',dict(complete=True,outer_loaded=False,goal_complete=False));print(json.dumps(dict(new=new,control=old,wins=wins,expand=expand)),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
