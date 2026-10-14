"""Independent replay of all pace-aligned shape candidates."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.tail_boost import inputs,conditions
from battery_smetimes.conditional_drift import prefix
from scripts.train_tail_boost import load_cells
from scripts.train_crossfit_drift_smoke import clean
from scripts.verify_neighbor_curve_nca_formal import reference_replay
from scripts.audit_tail_numeric_gates import metric
from scripts.prepare_same_type_forecast import sha,save_json


def validation_score(rows):
    v=np.array([(r['mape_percent']+r['late_mape']+r['endpoint_pp']+r['drop_error_pp'])/4 for r in rows]);return float(v.mean()+.25*np.quantile(v,.9))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text());assert json.loads((root/'progress.json').read_text())['complete']
    for f,h in cfg['source_sha256'].items():assert sha(f)==h and sha(root/'source_snapshot'/Path(f).name)==h
    parent=Path(cfg['parent']);assert sha(parent/'independent_verification.json')==cfg['parent_verification_sha256'] and sha(parent/'inner_lock.json')==cfg['inner_lock_sha256']
    assert cfg['outer_loaded'] is False and cfg['original_validation_loaded'] is False
    expected=[dict(mode='raw',k=1,weight=0.)]+[dict(mode=mode,k=k,weight=w) for mode in ('raw','shape') for k in (1,3) for w in (.25,.5,1.)]+[dict(mode='rate_shape',k=k,weight=w) for k in (1,3) for w in (.25,.5,1.)];assert cfg['candidates']==expected
    summary=json.loads((root/'summary.json').read_text());records=[];curves=0
    for lock in cfg['children']:
        child=Path(lock['inner_run']);cc=json.loads((child/'config.json').read_text());assert sha(child/'config.json')==lock['inner_config_sha256'] and sha(child/'independent_verification.json')==lock['verification_sha256']
        plan=cc['parent_plan'];cells=load_cells(dict(plan,validation=[]));assert set(cells)==set(plan['train']) and not set(cells)&set(plan['validation']+plan['test']);folder=root/plan['name'];pooled=[[] for _ in expected]
        anchors={(r['inner'],r['cell_id']):r for r in json.loads((folder/'anchors.json').read_text())};support={(r['inner'],r['cell_id']):r for r in json.loads((folder/'support.json').read_text())};assert len(anchors)==len(support)==39
        for split in cc['splits']:
            model_file=child/split['name']/'components.joblib';assert sha(model_file)==lock['component_sha256'][split['name']];parts=joblib.load(model_file);reference=reference_replay(parts['analog'],cells,split['train']);cached={}
            with np.load(child/split['name']/'validation_components.npz') as a:
                for key in split['validation']:
                    query=clean(cells[key]);primary=parts['primary']['inner'];start=float(primary['model'].predict(primary['imputer'].transform(inputs(query,np.array([101.]))))[0]);near={k:float(reference(query,np.array([101.]),k,None)[0]) for k in (1,3)}
                    saved=anchors[(split['name'],key)]
                    for f,v in [('primary',start),('reference1',near[1]),('reference3',near[3])]:np.testing.assert_allclose(v,saved[f],rtol=0,atol=1e-12)
                    c,y,p=[a[key+'__100__'+f] for f in ('cycle','truth','primary')];np.testing.assert_allclose(p,primary['model'].predict(primary['imputer'].transform(inputs(query,c))),rtol=0,atol=1e-12)
                    refs={k:reference(query,c,k,None) for k in (1,3)}
                    for k in (1,3):np.testing.assert_allclose(refs[k],a[key+'__100__neighbor'+str(k)],rtol=0,atol=1e-12)
                    bundle=parts['analog'];local=[i for i,t in enumerate(bundle['references']) if t['condition']==conditions(query)];pool=local if len(local)>=3 else list(range(len(bundle['references'])));z=bundle['scaler'].transform(bundle['imputer'].transform(prefix(query)[None]))[0];order=sorted(pool,key=lambda i:(float(np.linalg.norm(z-bundle['z'][i])),bundle['train_ids'][i]))[:3];s=support[(split['name'],key)]
                    assert s['protocol_peers']==len(local) and s['family_fallback']==(len(local)<3) and s['reference_ids']==[bundle['train_ids'][i] for i in order];np.testing.assert_allclose(s['distances'],[np.linalg.norm(z-bundle['z'][i]) for i in order],rtol=0,atol=1e-12)
                    end=float(primary['model'].predict(primary['imputer'].transform(inputs(query,np.array([201.]))))[0]);refend={k:float(reference(query,np.array([201.]),k,None)[0]) for k in (1,3)};ratios={k:(1. if start-end<=.001 or near[k]-refend[k]<=.001 else min(2.,max(.5,(start-end)/(near[k]-refend[k])))) for k in (1,3)}
                    for f,v in [('primary201',end),('reference1_201',refend[1]),('reference3_201',refend[3]),('pace1',ratios[1]),('pace3',ratios[3])]:np.testing.assert_allclose(v,saved[f],rtol=0,atol=1e-12)
                    warped={k:reference(query,101+ratios[k]*(c-101),k,None) for k in (1,3)};cached[key]=(c,y,p,refs,start,near,warped)
            for i,param in enumerate(expected):
                with np.load(folder/split['name']/f'candidate_{i}_predictions.npz') as a:
                    assert set(a.files)=={k+'__100__'+f for k in split['validation'] for f in ('cycle','truth','prediction')}
                    for key in split['validation']:
                        c,y,p,q,start,near,warped=cached[key];r=warped[param['k']] if param['mode']=='rate_shape' else q[param['k']];adjusted=r if param['mode']=='raw' else r-near[param['k']]+start;v=(1-param['weight'])*p+param['weight']*adjusted
                        for f,val in [('cycle',c),('truth',y),('prediction',v)]:np.testing.assert_allclose(val,a[key+'__100__'+f],rtol=0,atol=1e-12)
                        pooled[i].append(dict(cell_id=key,**metric(c,y,v),initial_signed_pp=float(100*(v[0]-y[0])),trend_signed_pp=float(100*((v[-1]-v[0])-(y[-1]-y[0])))));curves+=1
        record=json.loads((folder/'selection.json').read_text());scores=[]
        for i,rows in enumerate(pooled):
            stored=json.loads((folder/f'candidate_{i}_metrics.json').read_text());assert len(rows)==len(stored)==39
            for a,b in zip(rows,stored):
                assert a['cell_id']==b['cell_id']
                for f,v in a.items():
                    if f!='cell_id':np.testing.assert_allclose(v,b[f],rtol=0,atol=1e-10)
            result=dict(candidate=i,parameters=expected[i],score=validation_score(rows),**{f:float(np.mean([r[f] for r in rows])) for f in ('mape_percent','late_mape','endpoint_pp','drop_error_pp')},initial_abs_pp=float(np.mean([abs(r['initial_signed_pp']) for r in rows])),trend_abs_pp=float(np.mean([abs(r['trend_signed_pp']) for r in rows])));scores.append(result)
            for f,v in result.items():
                if f=='parameters':assert v==record['all_candidates'][i][f]
                else:np.testing.assert_allclose(v,record['all_candidates'][i][f],rtol=0,atol=1e-10)
        winner=min(scores,key=lambda r:(r['score'],r['candidate']));raw=min(scores[:13],key=lambda r:(r['score'],r['candidate']));assert record['selected']['candidate']==winner['candidate'] and record['raw_control']['candidate']==raw['candidate'];records.append(dict(selected=winner,raw_control=raw));print('VERIFIED '+plan['name'],flush=True)
    fields=('mape_percent','endpoint_pp','drop_error_pp');new={f:float(np.mean([r['selected'][f] for r in records])) for f in fields};old={f:float(np.mean([r['raw_control'][f] for r in records])) for f in fields};wins=sum(r['selected']['score']<r['raw_control']['score'] for r in records);expand=wins>=4 and new['endpoint_pp']<=old['endpoint_pp'] and new['drop_error_pp']<=old['drop_error_pp'] and new['mape_percent']<=old['mape_percent']+.1
    assert wins==summary['screen']['score_wins'] and expand==summary['screen']['expand']
    for label,actual in [('new',new),('raw_control',old)]:
        for f,v in actual.items():np.testing.assert_allclose(v,summary['screen'][label][f],rtol=0,atol=1e-10)
    save_json(root/'independent_verification.json',dict(passed=True,curves=curves,expand=expand,original_validation_loaded=False,outer_loaded=False,scope='training-only transformation replay; no new training or independent battery count',source_sha256=sha(__file__),goal_complete=False));print(json.dumps(dict(passed=True,curves=curves,expand=expand)),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
