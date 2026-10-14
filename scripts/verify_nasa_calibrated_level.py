"""Independent component lineage, fixed predicted anchors and offset replay."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.research_baselines import predict_baseline
from battery_smetimes.nasa_cycle_calibration import predict as calibrate
from battery_smetimes.nasa_calibrated_level import predict
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.audit_tail_numeric_gates import metric


def score(rows):
    v=np.array([(r['mape_percent']+r['late_mape']+r['endpoint_pp']+r['drop_error_pp'])/4 for r in rows]);return float(v.mean()+.25*np.quantile(v,.9))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text());assert json.loads((root/'progress.json').read_text())['complete'];parent=Path(cfg['parent']);assert sha(parent/'config.json')==cfg['parent_config_sha256'] and sha(parent/'independent_verification.json')==cfg['parent_verification_sha256'];assert json.loads((parent/'independent_verification.json').read_text())['passed']
    for f,h in cfg['source_sha256'].items():assert sha(f)==h and sha(root/'source_snapshot'/Path(f).name)==h
    records=[];curves=0;largest=0.
    for plan in cfg['plans']:
        cells=load_subset(plan['train']+plan['validation']);assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(plan['test']);folder=root/plan['name'];results=[]
        for i in range(13):
            dest=folder/f'candidate_{i}';bundle=joblib.load(dest/'model.joblib');assert bundle['train_ids']==plan['train'];lineage=json.loads((dest/'provenance.json').read_text());assert sha(lineage['primary_file'])==lineage['primary_sha256'];primary=joblib.load(lineage['primary_file']);assert joblib.hash(primary)==joblib.hash(bundle['primary']) and primary['train_ids']==plan['train']
            calibration=None
            if i:
                assert sha(lineage['calibration_file'])==lineage['calibration_sha256'];calibration=joblib.load(lineage['calibration_file']);assert joblib.hash(calibration)==joblib.hash(bundle['calibration']) and calibration['train_ids']==plan['train']
            else:assert bundle['calibration'] is None
            anchors={(r['role'],r['cell_id']):r for r in json.loads((dest/'anchors.json').read_text())};assert len(anchors)==3
            for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
                rows=[];stored=json.loads((dest/(role+'_metrics.json')).read_text());assert [r['cell_id'] for r in stored]==ids
                with np.load(dest/(role+'_predictions.npz')) as a:
                    assert set(a.files)=={key+'__100__'+f for key in ids for f in ('cycle','truth','prediction')}
                    for j,key in enumerate(ids):
                        query=clean(cells[key]);c=a[key+'__100__cycle'];y=a[key+'__100__truth'];base=predict_baseline(primary,query,100,c);b=float(predict_baseline(primary,query,100,[101.])[0]);v=float(calibrate(calibration,query,[101.])[0]) if i else b;p=base+v-b
                        for f,value in [('primary101',b),('calibrated101',v),('offset_soh',v-b)]:np.testing.assert_allclose(value,anchors[(role,key)][f],rtol=0,atol=1e-12)
                        np.testing.assert_allclose(p,a[key+'__100__prediction'],rtol=0,atol=1e-12);delta=abs((p[0]-p[-1])-(base[0]-base[-1]));largest=max(largest,float(delta));assert delta<1e-12
                        poison=dict(query,soh=np.inf,label_cycle=np.inf,reference_capacity_ah=np.inf,cycle_life=np.inf,cycle=np.r_[query['cycle'],101],raw=np.vstack([query['raw'],np.full((1,query['raw'].shape[1]),np.inf)]));np.testing.assert_allclose(p,predict(bundle,poison,c),rtol=0,atol=1e-12)
                        try:predict(bundle,dict(query,physical_family='wrong'),c)
                        except ValueError:pass
                        else:raise AssertionError('family gate absent')
                        keep=(cells[key]['label_cycle']>100)&np.isfinite(cells[key]['soh'])&(cells[key]['soh']>0);np.testing.assert_array_equal(c,cells[key]['label_cycle'][keep]);np.testing.assert_array_equal(y,cells[key]['soh'][keep]);r=dict(cell_id=key,**metric(c,y,p));rows.append(r);curves+=1
                        for f,value in r.items():
                            if f!='cell_id':np.testing.assert_allclose(value,stored[j][f],rtol=0,atol=1e-10)
                if role=='validation':results.append(dict(candidate=i,score=score(rows),**{f:float(np.mean([r[f] for r in rows])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')}))
            saved=json.loads((folder/'selection.json').read_text())['candidates'][i]
            for f,value in results[-1].items():np.testing.assert_allclose(value,saved[f],rtol=0,atol=1e-10)
        winner=min(results,key=lambda r:(r['score'],r['candidate']));stored=json.loads((folder/'selection.json').read_text());assert winner['candidate']==stored['selected']['candidate'];records.append(dict(selected=winner,control=results[0]));print('VERIFIED '+plan['name'],flush=True)
    fields=('score','mape_percent','endpoint_pp','drop_error_pp');new={f:float(np.mean([r['selected'][f] for r in records])) for f in fields};old={f:float(np.mean([r['control'][f] for r in records])) for f in fields};wins=sum(r['selected']['score']<r['control']['score'] for r in records);expand=wins>=3 and new['endpoint_pp']<=old['endpoint_pp']+1e-10 and new['drop_error_pp']<=old['drop_error_pp']+1e-10 and new['mape_percent']<=old['mape_percent']+.1;summary=json.loads((root/'summary.json').read_text());assert wins==summary['wins'] and expand==summary['expand']
    for label,actual in [('new',new),('control',old)]:
        for f,value in actual.items():np.testing.assert_allclose(value,summary[label][f],rtol=0,atol=1e-10)
    save_json(root/'independent_verification.json',dict(passed=True,curves=curves,max_drop_difference_soh=largest,expand=expand,outer_loaded=False,scope='prediction transformation and inherited independently verified models, no new model fits',source_sha256=sha(__file__),goal_complete=False));print(json.dumps(dict(passed=True,curves=curves,expand=expand)),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
