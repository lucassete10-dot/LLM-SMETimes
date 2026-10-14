"""Verified five-fold, five-seed N=100 MIT checkpoint for staged delivery."""
import json
from pathlib import Path
import numpy as np
from scripts.train_continuous_length_pilot import evaluate
from scripts.prepare_same_type_forecast import sha,save_json


def main():
    base=Path('experiments/core_length_cv')
    runs=[base/name for name in ['20261008T080847_529428Z','20261008T083248_563395Z',
        '20261008T083354_140849Z','20261008T083458_693830Z','20261008T083603_486376Z']]
    configs=[json.loads((r/'config.json').read_text()) for r in runs]
    assert [c['seeds'][0] for c in configs]==list(range(20261008,20261013))
    assert all(c['plans']==configs[0]['plans'] for c in configs)
    assert all(c['data_sha256']==configs[0]['data_sha256'] for c in configs)
    seed_rows=[];by_cell={};arrays={}
    for root,cfg in zip(runs,configs):
        proof=json.loads((root/'verification.json').read_text());assert proof['passed'] and proof['each_development_cell_validates_in_exactly_one_fold']
        rr=[r for r in json.loads((root/'all_rows.json').read_text()) if r['origin']==100 and r['arm']=='available' and r['horizon']=='operational']
        assert len(rr)==len({r['cell_id'] for r in rr})==107
        seed_rows.append(dict(seed=cfg['seeds'][0],**{m:float(np.mean([r[m] for r in rr])) for m in ['mape_percent','mae_soh','rmse_soh','late20_mae_pp','stage_error_pp','acceleration_direction_reversed']},mape_goal_cells=sum(r['mape_percent']<=6 for r in rr)))
        for rec in rr:by_cell.setdefault(rec['cell_id'],[]).append(rec)
        for plan in cfg['plans']:
            with np.load(root/plan['name']/'100'/'available'/'predictions.npz',allow_pickle=False) as data:
                for cid in plan['validation']:
                    c,y,p=[data[cid+'__'+k] for k in ['cycle','truth','prediction']]
                    if cid not in arrays:arrays[cid]=dict(cycle=c.copy(),truth=y.copy(),prediction=[])
                    else:
                        np.testing.assert_equal(arrays[cid]['cycle'],c);np.testing.assert_equal(arrays[cid]['truth'],y)
                    arrays[cid]['prediction'].append(p.copy())
    cells=[]
    for cid,rr in by_cell.items():
        a=arrays[cid];p=np.mean(a['prediction'],axis=0)
        cells.append(dict(cell_id=cid,seed_mean_mape=float(np.mean([r['mape_percent'] for r in rr])),
            seed_sd_mape=float(np.std([r['mape_percent'] for r in rr],ddof=1)),
            seed_mean_stage_pp=float(np.mean([r['stage_error_pp'] for r in rr])),
            mean_prediction_metrics=evaluate(a['cycle'],a['truth'],p)))
    metrics=list(seed_rows[0].keys())[1:-1]
    result=dict(dataset='MIT',origin=100,folds=5,seeds=5,cells=107,
        sources=[dict(run=str(r.resolve()),verification_sha256=sha(r/'verification.json')) for r in runs],
        mean_of_seed_cell_metrics={k:float(np.mean([r[k] for r in seed_rows])) for k in metrics},
        sample_seed_sd={k:float(np.std([r[k] for r in seed_rows],ddof=1)) for k in metrics},
        seed_rows=seed_rows,cell_rows=cells,
        mape_goal_cells_on_seed_mean=sum(c['seed_mean_mape']<=6 for c in cells),
        independent_test=False,goal_complete=False,shape_gate_complete=False,
        limitation='This is a developmental held-out-cell checkpoint, not independent generalization. Mean of seed errors differs from error of the mean forecast. No prediction confidence interval is claimed.',
        delivery_pending=['107 main curves','workbook','per-point prediction index','shape/knee diagnostics'])
    path=base/'MIT_five_seed_summary_20261008.json';save_json(path,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['cell_rows','sources','seed_rows']}))


if __name__=='__main__':main()
