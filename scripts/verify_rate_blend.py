"""Replay both audited components, every curve/metric and inner selection."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.rate_blend import make,predict
from battery_smetimes.fade_rate import predict as rate_predict
from battery_smetimes.tail_selection import choose,predict as primary_predict
from battery_smetimes.tail_boost import labels
from scripts.train_tail_boost import load_cells
from scripts.train_crossfit_drift_smoke import clean
from scripts.audit_tail_numeric_gates import metric
from scripts.prepare_same_type_forecast import save_json,sha


def score(rows):
    s=np.array([sum(r[f] for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp'))/4 for r in rows])
    return float(np.mean(s)+.25*np.quantile(s,.9))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text())
    assert json.loads((root/'progress.json').read_text())['assembly_complete'] and cfg['outer_loaded'] is False and cfg['weights']==[0,.25,.5,.75,1]
    source=Path(cfg['source_run']);route=Path(cfg['primary_run']);pcfg=json.loads((route/'config.json').read_text())
    assert sha(source/'config.json')==cfg['source_config_sha256'] and sha(route/'config.json')==cfg['primary_config_sha256']
    assert cfg['plans']==pcfg['plans']==json.loads((source/'config.json').read_text())['plans']
    assert json.loads((source/'independent_verification.json').read_text())['passed']
    assert sha('docs/soh_rate_blend_stage_20261005.md')==cfg['preregistration_sha256']
    for name,digest in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/name)==digest
    for name in ('rate_blend','fade_rate','tail_selection','tail_boost','curve_fidelity','forecast','research_baselines'):
        assert sha(Path('battery_smetimes')/(name+'.py'))==cfg['source_sha256'][name+'.py']
    reports=json.loads((root/'summary.json').read_text());checked=[];selected=[];old=[]
    for plan in cfg['plans']:
        cells=load_cells(plan);assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(plan['test'])
        for key in cells:assert sha(Path('data/interim/same_type_forecast/20260929T120631_212616Z')/(key+'.npz'))==cfg['prepared_sha256'][key]
        folder=root/plan['name'];prov=json.loads((folder/'provenance.json').read_text());parts=joblib.load(folder/'components.joblib');primary,rate=parts['primary'],parts['rate']
        decision=choose(route,pcfg,plan,cfg['seed'],cells);assert decision==prov['decision'];sel=decision['selected']
        pf=Path(sel['selection_file']).parent/f"candidate_{sel['candidate']}"/'model.joblib';rf=source/plan['name']/'candidate_1'/'model.joblib'
        assert Path(prov['primary_file'])==pf.resolve() and sha(pf)==prov['primary_sha256']==sel['model_sha256']
        assert Path(prov['rate_file'])==rf.resolve() and sha(rf)==prov['rate_sha256']
        assert primary['train_ids']==rate['train_ids']==primary['inner']['train_ids']==plan['train'] and rate['seed']==cfg['seed']
        replay_primary=dict(mode=sel['mode'],inner=joblib.load(pf),train_ids=plan['train'],family=rate['family']);replay_rate=joblib.load(rf)
        cache={}
        for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
            cache[role]={}
            with np.load(rf.parent/(role+'_predictions.npz')) as rate_arrays:
                for key in ids:
                    cell=clean(cells[key]);cycle,truth=labels(cells[key]);p=primary_predict(primary,cell,cycle);q=rate_predict(rate,cell,cycle)
                    np.testing.assert_allclose(p,primary_predict(replay_primary,cell,cycle),atol=1e-12,rtol=0);np.testing.assert_allclose(q,rate_predict(replay_rate,cell,cycle),atol=1e-12,rtol=0)
                    np.testing.assert_allclose(q,rate_arrays[key+'__100__prediction'],atol=1e-12,rtol=0)
                    np.testing.assert_array_equal(cycle,rate_arrays[key+'__100__cycle']);np.testing.assert_array_equal(truth,rate_arrays[key+'__100__truth'])
                    poison=dict(cell,cycle=np.r_[cell['cycle'],101],raw=np.vstack([cell['raw'],np.full((1,cell['raw'].shape[1]),np.inf)]),soh=np.inf,cycle_life=np.inf)
                    np.testing.assert_allclose(p,primary_predict(primary,poison,cycle),atol=1e-12,rtol=0);np.testing.assert_allclose(q,rate_predict(rate,poison,cycle),atol=1e-12,rtol=0)
                    try:predict(make(primary,rate,.5),dict(cell,physical_family='different'),cycle)
                    except ValueError:pass
                    else:raise AssertionError('family gate failed')
                    cache[role][key]=(cycle,truth,p,q)
        for index,w in enumerate(cfg['weights']):
            dest=folder/f'candidate_{index}';manifest=json.loads((dest/'model.json').read_text());assert manifest['weight']==w and manifest['train_ids']==plan['train'] and manifest['family']==rate['family']
            assert Path(manifest['components_file'])==(folder/'components.joblib').resolve() and sha(folder/'components.joblib')==manifest['components_sha256']
            for role,ids in [('training',plan['train']),('validation',plan['validation'])]:
                records={r['cell_id']:r for r in json.loads((dest/(role+'_metrics.json')).read_text())};assert set(records)==set(ids);values=[]
                with np.load(dest/(role+'_predictions.npz')) as arrays:
                    assert len(arrays.files)==3*len(ids)
                    for key,(cycle,truth,p,q) in cache[role].items():
                        prediction=(1-w)*p+w*q
                        for field,a in [('cycle',cycle),('truth',truth),('prediction',prediction)]:np.testing.assert_allclose(a,arrays[key+'__100__'+field],atol=1e-12,rtol=0)
                        m=metric(cycle,truth,prediction)
                        for f,v in m.items():np.testing.assert_allclose(v,records[key][f],atol=1e-10,rtol=0)
                        values.append(m);checked.append(dict(fold=plan['name'],candidate=index,role=role,cell_id=key,prediction_sha256=sha(dest/(role+'_predictions.npz'))))
                if role=='validation':val=values
            report=next(r for r in reports if r['fold']==plan['name'] and r['candidate']==index);np.testing.assert_allclose(score(val),report['score'],atol=1e-10,rtol=0)
            for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp'):np.testing.assert_allclose(np.mean([v[f] for v in val]),report[f],atol=1e-10,rtol=0)
        eligible=[r for r in reports if r['fold']==plan['name']];best=min(eligible,key=lambda r:(r['score'],r['weight']));assert best==json.loads((folder/'selection.json').read_text())['selected'];selected.append(best);old.append(eligible[0])
        print('PASS '+plan['name'],flush=True)
    expected=sum(len(p['train'])+len(p['validation']) for p in cfg['plans'])*len(cfg['weights']);assert len(checked)==expected
    save_json(root/'independent_verification.json',dict(passed=True,checked=checked,curves=expected,new_models_retrained=0,components_reused_from_independently_verified_runs=True,outer_loaded=False,source_sha256=sha(__file__),goal_complete=False))
    fields=('score','mape_percent','late_mape','endpoint_pp','drop_error_pp');means={name:{f:float(np.mean([r[f] for r in rr])) for f in fields} for name,rr in [('blend',selected),('primary',old)]}
    wins=sum(a['score']<b['score'] for a,b in zip(selected,old));passed=wins>=4 and means['blend']['score']<means['primary']['score'] and all(means['blend'][f]<=means['primary'][f] for f in ('endpoint_pp','drop_error_pp'))
    save_json(root/'screening_decision.json',dict(means=means,selected=selected,better_folds=wins,expand_formal=bool(passed),aggregation='one seed; inner validation; equal-cell within fold then equal-fold',outer_loaded=False,goal_complete=False))
    print('CURVES='+str(expected)+' SCREEN='+str(passed),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
