"""Aggregate complete cell-held-out length runs; fail on missing folds/cells."""
import json
from pathlib import Path
import numpy as np
from scripts.prepare_same_type_forecast import save_json


def main():
    root=Path('experiments/hust_length_cv'); paths=sorted(p for p in root.iterdir() if p.is_dir())
    records=[]; configs=[]
    for p in paths:
        proof=json.loads((p/'verification.json').read_text()); assert proof['passed']
        cfg=json.loads((p/'config.json').read_text()); configs.append(cfg)
        assert not json.loads((p/'coverage.json').read_text())[0]['excluded']
        assert not json.loads((p/'coverage.json').read_text())[1]['excluded']
        records+=json.loads((p/'all_rows.json').read_text())
    assert len(configs)==5 and set(c['cv_fold'] for c in configs)==set(range(1,6))
    dev=configs[0]['development_cells']; assert len(dev)==61
    assert sorted(k for c in configs for k in c['validation'])==dev
    assert all(c['development_cells']==dev and c['test']==configs[0]['test'] for c in configs)
    table=[]
    for origin in configs[0]['origins']:
        for candidate in configs[0]['candidates']:
            for horizon in ('common','operational'):
                rows=[r for r in records if r['origin']==origin and r['candidate']==candidate and r['horizon']==horizon]
                assert len(rows)==61 and sorted(r['cell_id'] for r in rows)==dev
                d=dict(origin=origin,candidate=candidate,horizon=horizon,cells=61)
                for metric in ('mape_percent','mae_soh','late20_mae_pp','stage_error_pp'):
                    values=np.array([r[metric] for r in rows]); d[metric]=float(values.mean())
                    d[metric+'_cell_sd']=float(values.std(ddof=1))
                d['numeric_pass_fraction']=float(np.mean([r['mape_percent']<=6 for r in rows]))
                d['reversal_fraction']=float(np.mean([r['acceleration_direction_reversed'] for r in rows]))
                rule=configs[0]['length_rule']
                d['qualifies_exploratory_band']=bool(d['mape_percent']<=rule['numeric_mape_goal'] and
                    d['late20_mae_pp']<=rule['late_mae_pp'] and d['stage_error_pp']<=rule['stage_pp'] and
                    d['reversal_fraction']<=rule['max_reversal_fraction'])
                table.append(d)
                if horizon=='common': print(origin,candidate,round(d['mape_percent'],3),round(d['late20_mae_pp'],3),round(d['stage_error_pp'],3),d['qualifies_exploratory_band'])
    save_json(root/'five_fold_summary.json',dict(runs=[str(p.resolve()) for p in paths],rows=table,
        unique_development_cells=61,folds=5,seeds=[20261008],independent_test=False,goal_complete=False,
        uncertainty_definition='Between-cell sample SD, NOT seed SD or confidence interval.',
        input_recommendation=None,reason='No final reliable recommendation until qualification and multi-seed confirmation.'))


if __name__=='__main__':main()
