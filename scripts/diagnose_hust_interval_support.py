"""Post-hoc development diagnostics; training-support evidence, not future inputs."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from battery_smetimes import voltage_interval_features as model,continuous_length as point
from scripts.prepare_same_type_forecast import sha,save_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);root=parser.parse_args().run
    cfg=json.loads((root/'config.json').read_text())
    assert json.loads((root/'verification.json').read_text())['passed']
    cells={}
    for pathkey in ['source_index','physics_index']:
        for rec in json.loads(Path(cfg[pathkey]).read_text())['cells']:
            cid=rec['cell_id']
            if cid not in cfg['train']+cfg['validation']:continue
            assert sha(Path(rec['path']))==rec['sha256']
            with np.load(rec['path'],allow_pickle=False) as d:cells.setdefault(cid,{}).update({k:d[k].copy() for k in d.files})
            if 'physical_family' in rec:cells[cid]['physical_family']=rec['physical_family']
    folder=root/'100'/'voltage_intervals';bundle=joblib.load(folder/'model.joblib')
    xs=[];ys=[];cycles=[];labels=[];weights=[]
    for cid in bundle['train_ids']:
        c,y=cells[cid]['cycle'],cells[cid]['truth'];take=(c>100)&np.isfinite(y)&(y>0);c,y=c[take],y[take]
        ii=np.unique(np.linspace(0,len(c)-1,min(96,len(c))).round().astype(int))
        xs.append(model.inputs(cells[cid],100,c[ii]));ys.extend(y[ii]);cycles.extend(c[ii]);labels.extend([cid]*len(ii));weights.extend([1/len(ii)]*len(ii))
    transform=lambda x:bundle['scaler'].transform(bundle['imputer'].transform(x))
    leaf=bundle['model'].apply(transform(np.concatenate(xs)));weights=np.array(weights);ys=np.array(ys);cycles=np.array(cycles);labels=np.array(labels)
    training_ends={cid:float(cells[cid]['cycle'][-1]) for cid in bundle['train_ids']}
    rows=[]
    with np.load(folder/'predictions.npz',allow_pickle=False) as forecasts:
        for cid in cfg['validation']:
            c,y,p=[forecasts[cid+'__'+k] for k in ['cycle','truth','prediction']]
            ii=np.unique(np.linspace(0,len(c)-1,4).round().astype(int))
            query_leaf=bundle['model'].apply(transform(model.inputs(cells[cid],100,c[ii])))
            probes=[]
            for j,index in enumerate(ii):
                membership=np.zeros(len(labels))
                for tree in range(leaf.shape[1]):
                    match=leaf[:,tree]==query_leaf[j,tree];total=weights[match].sum();assert total>0
                    membership[match]+=weights[match]/total/leaf.shape[1]
                np.testing.assert_allclose(sum(membership*ys),p[index],rtol=0,atol=1e-10)
                support={key:float(membership[labels==key].sum()) for key in bundle['train_ids']}
                top=sorted(support,key=support.get,reverse=True)[:3]
                probes.append(dict(source_cycle=float(c[index]),truth=float(y[index]),prediction=float(p[index]),
                    support_row_cycle_mean=float(sum(membership*cycles)),
                    support_rows_before_requested_fraction=float(membership[cycles<c[index]].sum()),
                    donor_end_cycle_mean=float(sum(support[key]*training_ends[key] for key in support)),
                    top_donors=[dict(cell_id=key,mass=support[key],last_training_cycle=training_ends[key]) for key in top]))
            rows.append(dict(cell_id=cid,diagnostic_probes=probes))
    importance=bundle['model'].feature_importances_
    original_width=len(point.features(cells[cfg['train'][0]],100))
    proof=dict(scope='Post-hoc development-only truth used for diagnosis, never a forecast feature.',
        query_time_feature_importance=float(sum(importance[-3:])),
        original_signal_feature_importance=float(sum(importance[:original_width])),
        interval_signal_feature_importance=float(sum(importance[original_width:-3])),
        prediction_reconstructed_from_training_leaf_weights=True,cells=rows,goal_complete=False)
    save_json(root/'interval_training_support.json',proof)
    print(json.dumps({k:v for k,v in proof.items() if k!='cells'}))
    for row in rows:
        if row['cell_id'] in ['HUST_7-7','HUST_3-8','HUST_2-7']:print(json.dumps(row))


if __name__=='__main__':main()
