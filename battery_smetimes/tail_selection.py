"""Frozen validation-only old/new choice and compatible family-gated refit."""
import json
from pathlib import Path
import joblib,numpy as np
from battery_smetimes.tail_boost import fit as new_fit,predict as new_predict,labels,is_nca
from battery_smetimes.curve_fidelity import fit_model,predict as core_predict,selection_score
from battery_smetimes.physical_types import load_registry
from scripts.train_nca_curve_fidelity import fit as nca_fit,predict as nca_predict
from scripts.train_curve_fidelity import AUDIT
from scripts.prepare_same_type_forecast import sha


def winner(new,original):
    """The preregistered rule: exact ties retain the frozen original."""
    return new if new['score']<original['score'] else original


def choose(run,cfg,plan,seed,cells):
    lock=json.loads((run/'reference_lock.json').read_text());assert lock['passed']
    ref=next(r for r in lock['references'] if r['fold']==plan['name'])
    root=Path(ref['run']);assert sha(root/'config.json')==ref['reference_config_sha256']
    oldcfg=json.loads((root/'config.json').read_text())
    oldplan=oldcfg['plan'] if ref['layout']=='core' else next(p for p in oldcfg['plans'] if p['name']==plan['name'])
    for role in ('train','validation','test'):assert oldplan[role]==plan[role]
    oldbase=root/str(seed) if ref['layout']=='core' else root/plan['name']/str(seed)
    oldsel=oldbase/'selection.json';assert sha(oldsel)==ref['selection_sha256'][str(seed)]
    old=json.loads(oldsel.read_text())['selected']
    newbase=run/plan['name']/str(seed);newsel=newbase/'selection.json'
    new=json.loads(newsel.read_text())['selected']
    modes=('tail','reference_core' if ref['layout']=='core' else 'reference_nca')
    choices=[]
    for mode,base,record in [(modes[0],newbase,new),(modes[1],oldbase,old)]:
        folder=base/f"candidate_{record['candidate']}"
        rows=json.loads((folder/'validation_metrics.json').read_text())
        assert {r['cell_id'] for r in rows}==set(plan['validation'])
        score=selection_score(rows)
        np.testing.assert_allclose(score,record['selection_score'],atol=1e-12,rtol=0)
        inner=joblib.load(folder/'model.joblib')
        with np.load(folder/'validation_predictions.npz') as arrays:
            for key in plan['validation']:
                c=cells[key];x,y=labels(c)
                np.testing.assert_array_equal(x,arrays[key+'__100__cycle']);np.testing.assert_array_equal(y,arrays[key+'__100__truth'])
                remove=('cycle','truth') if is_nca(c) else ('soh','label_cycle','unfiltered_soh','reference_capacity_ah')
                clean={k:v for k,v in c.items() if k not in remove}
                wrapper=dict(mode=mode,inner=inner,family=c['physical_family'])
                np.testing.assert_allclose(predict(wrapper,clean,x),arrays[key+'__100__prediction'],atol=1e-12,rtol=0)
        choices.append(dict(mode=mode,score=score,parameters=record['parameters'],candidate=record['candidate'],
            selection_file=str((base/'selection.json').resolve()),selection_sha256=sha(base/'selection.json'),
            model_sha256=sha(folder/'model.joblib')))
    selected=winner(choices[0],choices[1])
    return dict(selected=selected,candidates=choices,used_only_validation=True,ties_keep_original=True)


def fit(cells,ids,choice,seed):
    mode=choice['mode'];params=dict(choice['parameters'])
    if mode=='tail':inner=new_fit(cells,ids,params,seed)
    elif mode=='reference_nca':inner=nca_fit(cells,ids,params,seed)
    else:
        if params['arm'] in ('compact','ridge'):
            params['_registry']=load_registry(AUDIT/'source_snapshot/battery_physical_registry.json')
        inner=fit_model(cells,ids,seed,params)
    return dict(mode=mode,inner=inner,train_ids=list(ids),family=cells[ids[0]]['physical_family'],parameters=choice['parameters'])


def predict(bundle,cell,future):
    if cell['physical_family']!=bundle['family']:raise ValueError('different physical family')
    if bundle['mode']=='tail':return new_predict(bundle['inner'],cell,future)
    if bundle['mode']=='reference_nca':return nca_predict(bundle['inner'],cell,future)
    return core_predict(bundle['inner'],cell,future)
