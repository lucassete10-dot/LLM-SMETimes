"""Independent curve reference reconstruction; locked outer refits and arithmetic."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from battery_smetimes.conditional_drift import prefix
from battery_smetimes.tail_boost import labels,conditions
from battery_smetimes.tail_selection import choose,fit as primary_fit,predict as primary_predict
from scripts.train_tail_boost import load_cells
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.audit_rate_blend_metrics import metrics,summarize,checks,assert_summary

def reference_replay(bundle,cells,ids):
    assert bundle['train_ids']==ids and len(bundle['references'])==len(ids)
    assert {cells[k]['physical_family'] for k in ids}=={bundle['family']}
    raw=np.stack([prefix(clean(cells[k])) for k in ids]);imputer=SimpleImputer(strategy='median',keep_empty_features=True).fit(raw);scale=StandardScaler().fit(imputer.transform(raw));z=scale.transform(imputer.transform(raw))
    np.testing.assert_array_equal(imputer.statistics_,bundle['imputer'].statistics_);np.testing.assert_array_equal(scale.mean_,bundle['scaler'].mean_);np.testing.assert_array_equal(scale.scale_,bundle['scaler'].scale_);np.testing.assert_array_equal(z,bundle['z'])
    for record,key in zip(bundle['references'],ids):
        c,y=labels(cells[key]);assert record['cell_id']==key and record['condition']==conditions(cells[key]);np.testing.assert_array_equal(record['cycle'],c);np.testing.assert_array_equal(record['truth'],y)
    def predict(query,future,k,exclude):
        eligible=[i for i,key in enumerate(ids) if key!=exclude];local=[i for i in eligible if conditions(cells[ids[i]])==conditions(query)];pool=local if len(local)>=3 else eligible
        q=scale.transform(imputer.transform(prefix(query)[None]))[0];order=sorted(pool,key=lambda i:(float(np.linalg.norm(q-z[i])),ids[i]))[:k];values=[]
        for i in order:
            x,t=labels(cells[ids[i]]);p=np.interp(future,x,t);n=min(20,len(x))
            for mask,xx,yy,c0,y0 in [(future<x[0],x[:n],t[:n],x[0],t[0]),(future>x[-1],x[-n:],t[-n:],x[-1],t[-1])]:
                slope=np.polyfit(xx-xx.mean(),yy,1)[0];p[mask]=y0+slope*(future[mask]-c0)
            values.append(p)
        return np.mean(values,axis=0)
    return predict

def score(rows):
    values=np.array([(r['mape_percent']+r['late_mape']+r['endpoint_pp']+r['drop_error_pp'])/4 for r in rows]);return float(values.mean()+.25*np.quantile(values,.9))

def audit(folder,parts,primary,reference,cells,ids,role,params,exclude,seed):
    cached={};component_file=folder/(role+'_components.npz')
    with np.load(component_file) as a:
        assert set(a.files)=={key+'__100__'+f for key in ids for f in ('cycle','truth','primary','neighbor1','neighbor3')}
        for key in ids:
            c,y=labels(cells[key]);query=clean(cells[key]);p=primary_predict(primary,query,c);np.testing.assert_allclose(p,primary_predict(parts['primary'],query,c),rtol=0,atol=1e-12)
            q={k:reference(query,c,k,key if exclude else None) for k in (1,3)}
            for f,v in [('cycle',c),('truth',y),('primary',p),('neighbor1',q[1]),('neighbor3',q[3])]:np.testing.assert_allclose(v,a[key+'__100__'+f],rtol=0,atol=1e-12)
            poison=dict(query,truth=np.inf,soh=np.inf,cycle_life=np.inf,early_cycle=np.r_[query['early_cycle'],101],early_raw=np.vstack([query['early_raw'],np.full((1,query['early_raw'].shape[1]),np.inf)]))
            from battery_smetimes.neighbor_curve import predict as deployed
            for k in (1,3):np.testing.assert_allclose(q[k],deployed(parts['analog'],poison,c,k,exclude=key if exclude else None),rtol=0,atol=1e-12)
            np.testing.assert_allclose(p,primary_predict(parts['primary'],poison,c),rtol=0,atol=1e-12);cached[key]=(c,y,p,q)
    output=[]
    for index,param in enumerate(params):
        dest=folder/f'candidate_{index}' if folder.name=='inner' else folder;rows=[];stored=json.loads((dest/(role+'_metrics.json')).read_text());assert [r['cell_id'] for r in stored]==ids
        with np.load(dest/(role+'_predictions.npz')) as a:
            assert set(a.files)=={key+'__100__'+f for key in ids for f in ('cycle','truth','prediction')}
            for j,key in enumerate(ids):
                c,y,p,q=cached[key];v=(1-param['weight'])*p+param['weight']*q[param['k']]
                for f,value in [('cycle',c),('truth',y),('prediction',v)]:np.testing.assert_allclose(value,a[key+'__100__'+f],rtol=0,atol=1e-12)
                r=metrics(c,y,v)
                for f,value in r.items():np.testing.assert_allclose(value,stored[j][f],rtol=0,atol=1e-10)
                rows.append(dict(cell_id=key,seed=seed,**r))
        output.append(rows)
    return output

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text());progress=json.loads((root/'progress.json').read_text());assert progress['training_complete']
    for f,h in cfg['source_sha256'].items():assert sha(f)==h and sha(root/'source_snapshot'/Path(f).name)==h
    route=Path(cfg['training_source']);pcfg=json.loads((route/'config.json').read_text());assert sha(route/'config.json')==cfg['training_config_sha256'] and pcfg['plans']==cfg['plans'];assert cfg['seeds']==list(range(20261005,20261010))
    assert sha(root/'inner_lock.json')==progress['inner_lock_sha256'];locks=json.loads((root/'inner_lock.json').read_text())['selections'];assert len(locks)==25
    all_ids=[k for p in cfg['plans'] for k in p['test']];assert len(all_ids)==len(set(all_ids))==55
    lock_time=(root/'inner_lock.json').stat().st_mtime_ns;count=0;rows=[];refits=0
    assert len({(r['fold'],r['seed']) for r in locks})==25
    for plan in cfg['plans']:
        assert not set(plan['train'])&set(plan['validation']) and not set(plan['train']+plan['validation'])&set(plan['test'])
        inner_cells=load_cells(plan)
        for seed in cfg['seeds']:
            lock=next(r for r in locks if r['fold']==plan['name'] and r['seed']==seed);folder=root/plan['name']/str(seed)/'inner';assert sha(folder/'components.joblib')==lock['components_sha256'] and sha(folder/'selection.json')==lock['selection_sha256']
            assert (folder/'selection.json').stat().st_mtime_ns<=lock_time and (folder/'components.joblib').stat().st_mtime_ns<=lock_time
            parts=joblib.load(folder/'components.joblib');decision=choose(route,pcfg,plan,seed,inner_cells);assert decision['selected']==lock['primary_choice'];prov=json.loads((folder/'provenance.json').read_text());assert sha(prov['primary_file'])==prov['primary_sha256'];replay=dict(parts['primary'],inner=joblib.load(prov['primary_file']))
            reference=reference_replay(parts['analog'],inner_cells,plan['train']);audit(folder,parts,replay,reference,inner_cells,plan['train'],'training',cfg['candidates'],True,seed);rr=audit(folder,parts,replay,reference,inner_cells,plan['validation'],'validation',cfg['candidates'],False,seed);count+=7*(len(plan['train'])+len(plan['validation']))
            selected=min(range(7),key=lambda i:(score(rr[i]),i));assert selected==lock['selected']['candidate'];np.testing.assert_allclose(score(rr[selected]),lock['selected']['score'],rtol=0,atol=1e-10)
            print('INNER VERIFIED '+plan['name']+' '+str(seed),flush=True)
        ids=plan['train']+plan['validation'];cells=load_cells(dict(plan,train=ids+plan['test'],validation=[]))
        for seed in cfg['seeds']:
            lock=next(r for r in locks if r['fold']==plan['name'] and r['seed']==seed);folder=root/plan['name']/str(seed)/'outer';assert (folder/'model.joblib').stat().st_mtime_ns>=lock_time
            parts=joblib.load(folder/'model.joblib');assert parts['parameters']==lock['selected']['parameters'];reference=reference_replay(parts['analog'],cells,ids);replay=primary_fit(cells,ids,lock['primary_choice'],seed);assert parts['primary']['train_ids']==ids;refits+=1
            audit(folder,parts,replay,reference,cells,ids,'training',[parts['parameters']],True,seed);rr=audit(folder,parts,replay,reference,cells,plan['test'],'test',[parts['parameters']],False,seed)[0];rows.extend(rr);count+=len(ids)+len(plan['test']);print('OUTER VERIFIED '+plan['name']+' '+str(seed),flush=True)
    stored=json.loads((root/'summary.json').read_text());new,cellrows=summarize(rows);assert_summary(new,stored['new']);summaries={};evidence=[]
    for cohort,path,seeds,suffix in [('baseline',Path(cfg['baseline']),range(20261004,20261009),'predictions.npz'),('current_primary',Path(cfg['current_primary']),cfg['seeds'],'test_predictions.npz')]:
        other=[]
        for plan in cfg['plans']:
            for seed in seeds:
                file=path/plan['name']/str(seed)/suffix
                with np.load(file) as a,np.load(root/plan['name']/str(cfg['seeds'][0])/'outer/test_predictions.npz') as truth:
                    for key in plan['test']:
                        c,y,p=[a[key+'__100__'+f] for f in ('cycle','truth','prediction')];np.testing.assert_array_equal(c,truth[key+'__100__cycle']);np.testing.assert_array_equal(y,truth[key+'__100__truth']);other.append(dict(cell_id=key,seed=seed,**metrics(c,y,p)))
                evidence.append(dict(file=str(file.resolve()),sha256=sha(file)))
        summary,_=summarize(other);assert_summary(summary,stored[cohort]);summaries[cohort]=summary
    gate=checks(new,summaries['baseline']);assert gate==stored['gates'] and all(gate.values())==stored['numeric_gates_passed']
    bycell=json.loads((root/'by_cell.json').read_text());assert len(bycell)==55
    for a,b in zip(cellrows,bycell):
        assert a['cell_id']==b['cell_id']
        for f,v in a.items():
            if f!='cell_id':np.testing.assert_allclose(v,b[f],rtol=0,atol=1e-10)
    save_json(root/'independent_verification.json',dict(passed=True,curves=count,outer_primary_refits=refits,primary_refit_scope='same frozen primary fit implementation, freshly trained; reference algorithm independently reconstructed',new=new,gates=gate,numeric_gates_passed=all(gate.values()),failed_joint_cells=[c for c in cellrows if not c['joint_pass']],baseline_evidence=evidence,source_sha256=sha(__file__),goal_complete=False));print(json.dumps(dict(passed=True,curves=count,gates=gate)),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
