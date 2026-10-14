"""Independent replay of frozen component manifests and relative-curve algebra."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.pooled_anchor import anchor
from battery_smetimes.fade_rate import predict as fade
from battery_smetimes.tail_boost import labels
from scripts.train_curve_fidelity import load_subset
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import save_json,sha
from scripts.audit_tail_numeric_gates import metric


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text())
    assert json.loads((root/'assembly_audit.json').read_text())['passed']
    assert sha('docs/soh_decoupled_anchor_stage_20261005.md')==cfg['preregistration_sha256']
    pooled=Path(cfg['pooled_run']);rate=Path(cfg['rate_run']);assert sha(pooled/'config.json')==cfg['pooled_config_sha256'] and sha(rate/'config.json')==cfg['rate_config_sha256']
    assert json.loads((pooled/'independent_verification.json').read_text())['passed'] and json.loads((rate/'independent_verification.json').read_text())['passed']
    for name,digest in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/name)==digest
    reports=json.loads((root/'summary.json').read_text());checked=[]
    for plan in cfg['plans']:
        cells=load_subset(plan['train']+plan['validation']);assert not set(cells)&set(plan['test'])
        for index,params in enumerate(cfg['candidates']):
            folder=root/plan['name']/('candidate_'+str(index));manifest=json.loads((folder/'model.json').read_text());assert manifest['train_ids']==plan['train']
            original_rate=rate/plan['name']/str(cfg['seed'])/'inner'/'model.joblib';assert Path(manifest['rate_file'])==original_rate.resolve() and sha(original_rate)==manifest['rate_sha256'];base=joblib.load(original_rate)
            assert base['train_ids']==plan['train'] and not base['learned_anchor']
            model=None
            if params is not None:
                original_anchor=pooled/plan['name']/('candidate_'+str(index))/'model.joblib';assert Path(manifest['pooled_file'])==original_anchor.resolve() and sha(original_anchor)==manifest['pooled_sha256'];model=joblib.load(original_anchor)
                assert model['train_ids']==plan['train'] and model['family']==base['family'] and model['seed']==base['seed']
            vals=[]
            for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
                rows={r['cell_id']:r for r in json.loads((folder/(role+'_metrics.json')).read_text())};assert set(rows)==set(ids)
                with np.load(folder/(role+'_predictions.npz')) as arrays:
                    assert len(arrays.files)==3*len(ids)
                    for key in ids:
                        c=clean(cells[key]);x,y=labels(cells[key]);d=fade(base,c,x);p=d if model is None else anchor(model,c)+d-fade(base,c,np.array([101.]))[0]
                        np.testing.assert_array_equal(x,arrays[key+'__100__cycle']);np.testing.assert_array_equal(y,arrays[key+'__100__truth']);np.testing.assert_allclose(p,arrays[key+'__100__prediction'],atol=1e-12,rtol=0)
                        np.testing.assert_allclose(np.diff(p),np.diff(d),atol=1e-12,rtol=0)
                        v=metric(x,y,p)
                        for field,value in v.items():np.testing.assert_allclose(value,rows[key][field],atol=1e-10,rtol=0)
                        if role=='validation':vals.append(v)
                        checked.append(dict(fold=plan['name'],candidate=index,role=role,cell_id=key,prediction_sha256=sha(folder/(role+'_predictions.npz'))))
            scores=[sum(v[f] for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp'))/4 for v in vals];report=next(r for r in reports if r['fold']==plan['name'] and r['candidate']==index)
            np.testing.assert_allclose(np.mean(scores)+.25*np.quantile(scores,.9),report['score'],atol=1e-10,rtol=0)
    assert len(checked)==60;save_json(root/'independent_verification.json',dict(passed=True,checked=checked,new_models_trained=0,source_sha256=sha(__file__),outer_loaded=False,goal_complete=False));print('AUDIT_PASS curves=60',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
