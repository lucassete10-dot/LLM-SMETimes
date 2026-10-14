"""Independent nearest-reference selection/interpolation and all-curve replay."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from battery_smetimes.tail_boost import labels,conditions
from battery_smetimes.conditional_drift import prefix
from battery_smetimes.tail_selection import predict as primary_predict
from battery_smetimes.curve_fidelity import curve_metrics,selection_score
from scripts.train_tail_boost import load_cells
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import save_json,sha

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text());curves=0
    for f,h in cfg['source_sha256'].items():assert sha(f)==h
    for route in cfg['routes']:assert sha(Path(route['path'])/'config.json')==route['sha256']
    for plan in cfg['plans']:
        assert not set(plan['train'])&set(plan['validation']) and not set(plan['train']+plan['validation'])&set(plan['test'])
        cells=load_cells(plan);assert set(cells)==set(plan['train']+plan['validation']);folder=root/plan['name'];parts=joblib.load(folder/'components.joblib');b=parts['analog'];prov=json.loads((folder/'provenance.json').read_text());assert sha(prov['primary_file'])==prov['primary_sha256'];assert b['train_ids']==plan['train'] and parts['primary']['inner']['train_ids']==plan['train']
        source=joblib.load(prov['primary_file']);assert parts['primary']['mode']==prov['decision']['selected']['mode']
        raw=np.stack([prefix(clean(cells[k])) for k in plan['train']]);imp=SimpleImputer(strategy='median',keep_empty_features=True).fit(raw);scale=StandardScaler().fit(imp.transform(raw));z=scale.transform(imp.transform(raw))
        np.testing.assert_array_equal(imp.statistics_,b['imputer'].statistics_);np.testing.assert_array_equal(scale.mean_,b['scaler'].mean_);np.testing.assert_array_equal(scale.scale_,b['scaler'].scale_);np.testing.assert_array_equal(z,b['z'])
        refs=[]
        for r,key in zip(b['references'],plan['train']):
            c,y=labels(cells[key]);assert r['cell_id']==key and r['condition']==conditions(cells[key]);np.testing.assert_array_equal(r['cycle'],c);np.testing.assert_array_equal(r['truth'],y);refs.append((c,y))
        scores=[]
        results={role:[[] for _ in cfg['candidates']] for role in ('training','validation')}
        for role in ('training','validation'):
            ids=plan['train'] if role=='training' else plan['validation']
            saved=np.load(folder/(role+'_components.npz'))
            try:
                assert set(saved.files)=={k+'__100__'+f for k in ids for f in ('cycle','truth','primary','neighbor1','neighbor3')}
                candidate_arrays=[np.load(folder/f'candidate_{i}'/(role+'_predictions.npz')) for i in range(len(cfg['candidates']))]
                try:
                    for key in ids:
                        c,y=labels(cells[key]);query=clean(cells[key]);q=scale.transform(imp.transform(prefix(query)[None]))[0];eligible=[i for i,k in enumerate(plan['train']) if role!='training' or k!=key];local=[i for i in eligible if conditions(cells[plan['train'][i]])==conditions(query)];pool=local if len(local)>=3 else eligible;order=sorted(pool,key=lambda i:(float(np.linalg.norm(q-z[i])),plan['train'][i]))
                        base=primary_predict(dict(parts['primary'],inner=source),query,c);components={}
                        for k in (1,3):
                            values=[]
                            for i in order[:k]:
                                x,t=refs[i];p=np.interp(c,x,t);n=min(20,len(x))
                                for mask,xx,yy,anchor_x,anchor_y in [(c<x[0],x[:n],t[:n],x[0],t[0]),(c>x[-1],x[-n:],t[-n:],x[-1],t[-1])]:
                                    slope=np.polyfit(xx-xx.mean(),yy,1)[0];p[mask]=anchor_y+slope*(c[mask]-anchor_x)
                                values.append(p)
                            components[k]=np.mean(values,axis=0);np.testing.assert_allclose(components[k],saved[key+'__100__neighbor'+str(k)],rtol=0,atol=1e-12)
                        np.testing.assert_array_equal(c,saved[key+'__100__cycle']);np.testing.assert_array_equal(y,saved[key+'__100__truth']);np.testing.assert_allclose(base,saved[key+'__100__primary'],rtol=0,atol=1e-12)
                        for i,params in enumerate(cfg['candidates']):
                            a=candidate_arrays[i];p=(1-params['weight'])*base+params['weight']*components[params['k']];np.testing.assert_array_equal(a[key+'__100__cycle'],c);np.testing.assert_array_equal(a[key+'__100__truth'],y);np.testing.assert_allclose(p,a[key+'__100__prediction'],rtol=0,atol=1e-12);results[role][i].append(dict(cell_id=key,**curve_metrics(c,y,p)));curves+=1
                finally:
                    for a in candidate_arrays:a.close()
            finally:saved.close()
            for i,rows in enumerate(results[role]):
                old=json.loads((folder/f'candidate_{i}'/(role+'_metrics.json')).read_text());assert [r['cell_id'] for r in rows]==[r['cell_id'] for r in old]
                for r,t in zip(rows,old):
                    for field,value in r.items():
                        if field!='cell_id':np.testing.assert_allclose(value,t[field],rtol=0,atol=1e-10)
                if role=='validation':scores.append(dict(candidate=i,score=selection_score(rows),endpoint_pp=float(np.mean([r['endpoint_pp'] for r in rows])),drop_error_pp=float(np.mean([r['drop_error_pp'] for r in rows]))))
        selected=min(scores,key=lambda r:(r['score'],r['candidate']));record=json.loads((folder/'selection.json').read_text())
        for field,value in selected.items():np.testing.assert_allclose(value,record['selected'][field],rtol=0,atol=1e-10)
        control=scores[0];screen=selected['score']<control['score'] and selected['endpoint_pp']<=control['endpoint_pp'] and selected['drop_error_pp']<=control['drop_error_pp'];assert screen==record['expand_fivefold']
    save_json(root/'independent_verification.json',dict(passed=True,curves=curves,outer_loaded=False,preprocessing_independently_refit=True,reference_selection_and_extrapolation_independently_replayed=True,primary_scope='source hash and exact replay, not newly refit',source_sha256=sha(__file__),goal_complete=False));print(json.dumps(dict(passed=True,curves=curves)),flush=True)

if __name__=='__main__':main()
