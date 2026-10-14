"""Quantify training support at requested horizons without altering predictions."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from battery_smetimes import condition_length as local
from scripts.train_continuous_length_pilot import load_nominal, target
from scripts.prepare_same_type_forecast import save_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);root=ap.parse_args().run
    assert json.loads((root/'independent_replay_verification.json').read_text())['passed']
    cfg=json.loads((root/'config.json').read_text());cells,_=load_nominal(cfg['plan'])
    targets={k:target(c,100) for k,c in cells.items()};records=[]
    for origin in cfg['origins']:
        seed=cfg['seeds'][0]
        for name in ('global_point','local_point','local_rate'):
            b=joblib.load(root/str(origin)/str(seed)/name/'model.joblib')
            ids=json.loads((root/str(origin)/str(seed)/name/'config.json').read_text())['validation']
            for k in ids:
                child,matched=local.route(b,cells[k]);c,y=targets[k]
                ends=np.array([targets[t][0][-1] for t in child['train_ids']])
                p=local.predict(b,cells[k],c)
                late=c>=c[0]+.8*(c[-1]-c[0])
                records.append(dict(origin=origin,candidate=name,cell_id=k,used_local=matched,
                    donor_cells=[dict(cell_id=t,record_end=float(targets[t][0][-1])) for t in child['train_ids']],
                    requested_end=float(c[-1]),max_training_end=float(ends.max()),
                    fraction_grid_beyond_all_train_support=float(np.mean(c>ends.max())),
                    donors_covering_end=int(np.sum(ends>=c[-1])),
                    late_decline_prediction_pp=float(100*(p[late][0]-p[-1])),
                    late_decline_truth_pp=float(100*(y[late][0]-y[-1]))))
    save_json(root/'training_support_audit.json',dict(records=records,diagnostic_seed=cfg['seeds'][0],
        note='Record ends diagnose support; they are not test inputs or battery lifetime labels.'))
    for r in records:
        if r['cell_id'].startswith('b5_') and r['origin']==100:
            print(json.dumps({k:v for k,v in r.items() if k!='donor_cells'}))


if __name__=='__main__':main()
