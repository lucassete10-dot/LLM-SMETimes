"""Explain saved MIT forecasts by exact weighted training-leaf contributions."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from battery_smetimes import continuous_length as model
from scripts.train_core_length_cv import load
from scripts.train_continuous_length_pilot import target
from scripts.prepare_same_type_forecast import save_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);root=ap.parse_args().run
    cfg=json.loads((root/'config.json').read_text());cells=load(cfg['plans'][0]);expanded={}
    cache=Path(cfg['cache']) if cfg['cache'] else root
    for cid,cell in cells.items():
        with np.load(cache/(cid+'.features.npz')) as d:
            expanded[cid]=dict(cell,early_cycle=d['cycle'].copy(),early_raw=d['raw'].copy(),conditions=d['conditions'].copy())
    rows=[]
    for cid in ['b1c1','b1c4','b1c21','b1c29']:
        plan=next(p for p in cfg['plans'] if cid in p['validation'])
        for arm,src in [('prefix300',cells),('completed_past',expanded)]:
            folder=root/plan['name']/'100'/str(cfg['seeds'][0])/arm
            b=joblib.load(folder/'model.joblib');xs=[];ys=[];cs=[];ids=[];ws=[]
            for key in b['train_ids']:
                c,y=target(cells[key],100);ii=np.unique(np.linspace(0,len(c)-1,min(96,len(c))).round().astype(int))
                xs.append(model.point_inputs(src[key],100,c[ii],'VIT'));ys.extend(y[ii]);cs.extend(c[ii]);ids.extend([key]*len(ii));ws.extend([1/len(ii)]*len(ii))
            ys,cs,ids,ws=map(np.asarray,(ys,cs,ids,ws))
            transform=lambda x:b['scaler'].transform(b['imputer'].transform(x))
            leaves=b['model'].apply(transform(np.concatenate(xs)))
            with np.load(folder/'predictions.npz') as d:
                c,y,p=[d[cid+'__'+key].copy() for key in ('cycle','truth','prediction')]
            ii=np.unique(np.linspace(0,len(c)-1,5).round().astype(int))
            query=b['model'].apply(transform(model.point_inputs(src[cid],100,c[ii],'VIT')))
            probes=[]
            for j,index in enumerate(ii):
                membership=np.zeros(len(ids))
                for tree in range(leaves.shape[1]):
                    mask=leaves[:,tree]==query[j,tree]
                    membership[mask]+=ws[mask]/ws[mask].sum()/leaves.shape[1]
                np.testing.assert_allclose(membership@ys,p[index],atol=1e-10,rtol=0)
                support={key:float(membership[ids==key].sum()) for key in b['train_ids']}
                top=sorted(support,key=support.get,reverse=True)[:5]
                probes.append(dict(cycle=float(c[index]),truth=float(y[index]),prediction=float(p[index]),
                    supported_training_cycle_mean=float(membership@cs),
                    mass_from_earlier_cycles=float(membership[cs<c[index]-20].sum()),
                    donors=[dict(cell_id=key,mass=support[key],last_label_cycle=float(target(cells[key],100)[0][-1])) for key in top]))
            rows.append(dict(cell_id=cid,arm=arm,query_cycle_importance=float(b['model'].feature_importances_[-3:].sum()),probes=probes))
    save_json(root/'diagnostic_support.json',dict(exact_leaf_reconstruction_passed=True,
        scope='Post-hoc development diagnosis. Future truth and donor endings are never forecast inputs.',cells=rows))
    print('Exact leaf reconstruction passed for',len(rows),'cell/arm pairs')


if __name__=='__main__':main()
