"""Explain nonlinear parameter averaging using exact ExtraTrees leaf weights."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from battery_smetimes.continuous_length import features,basis,predict
from scripts.train_continuous_length_pilot import load_nominal,target
from scripts.prepare_same_type_forecast import save_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);root=ap.parse_args().run
    cfg=json.loads((root/'config.json').read_text());reports=[]
    for plan in cfg['plans']:
        cells,_=load_nominal(plan)
        folder=root/plan['name']/'100'/str(cfg['seeds'][0])/'continuous_V_leaf1'
        b=joblib.load(folder/'model.joblib');ids=b['train_ids']
        x=np.stack([features(cells[k],100,'V') for k in ids]);z=b['scaler'].transform(b['imputer'].transform(x))
        trainleaf=b['model'].apply(z);params=np.array([r['parameters'] for r in b['training_shape_fits']])
        for k in json.loads((folder/'config.json').read_text())['validation']:
            xx=features(cells[k],100,'V')[None];zz=b['scaler'].transform(b['imputer'].transform(xx));leaves=b['model'].apply(zz)[0]
            w=np.zeros(len(ids))
            for j,leaf in enumerate(leaves):
                hit=trainleaf[:,j]==leaf;assert hit.any();w[hit]+=1/hit.sum()/len(leaves)
            inferred=b['model'].predict(zz)[0]*b['target_scale']
            np.testing.assert_allclose(w@params,inferred,rtol=0,atol=1e-10)
            c,y=target(cells[k],100);actual=predict(b,cells[k],c)
            average=np.zeros_like(c);donors=[]
            for i,weight in enumerate(w):
                if not weight:continue
                coef=params[i];curve=basis(c,100,coef[3])@coef[:3];average+=weight*curve
                trainc,_=target(cells[ids[i]],100)
                donors.append(dict(cell_id=ids[i],weight=float(weight),parameters=coef.tolist(),train_last_cycle=float(trainc[-1]),
                    requested_beyond_training_span=bool(c[-1]>trainc[-1]),weighted_endpoint=float(weight*curve[-1])))
            reports.append(dict(dataset=plan['dataset'],cell_id=k,mean_parameter_curve_mape=100*float(np.mean(abs(actual-y)/y)),
                mean_donor_curve_mape=100*float(np.mean(abs(average-y)/y)),
                nonlinear_averaging_difference_endpoint_pp=100*float(actual[-1]-average[-1]),
                extrapolated_donor_weight=float(sum(d['weight'] for d in donors if d['requested_beyond_training_span'])),
                exact_leaf_weight_reconstruction=True,donors=sorted(donors,key=lambda d:-d['weight'])))
    save_json(root/'continuous_donor_audit.json',dict(reports=reports,validation_diagnostic_only=True,goal_complete=False))
    for r in reports:
        if r['dataset']=='XJTU':print(json.dumps({k:v for k,v in r.items() if k!='donors'}))


if __name__=='__main__':main()
