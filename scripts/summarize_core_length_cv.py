"""Cell-weighted length comparison; separate matched and actual-service coverage."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);root=parser.parse_args().run
    proof=json.loads((root/'verification.json').read_text());assert proof['passed']
    cfg=json.loads((root/'config.json').read_text());rows=json.loads((root/'all_rows.json').read_text())
    summary=[];recommendations=[]
    for arm in cfg['arms']:
        for n in cfg['origins']:
            for horizon in ['operational','common','fixed_50','fixed_100','fixed_200']:
                rr=[r for r in rows if r['arm']==arm and r['origin']==n and r['horizon']==horizon]
                assert len(rr)==len({r['cell_id'] for r in rr})
                if not rr:continue
                summary.append(dict(arm=arm,origin=n,horizon=horizon,cells=len(rr),development_cells=len(cfg['development']),
                    mape=float(np.mean([r['mape_percent'] for r in rr])),mape_cell_sd=float(np.std([r['mape_percent'] for r in rr],ddof=1)),
                    mape_median=float(np.median([r['mape_percent'] for r in rr])),mape_p90=float(np.percentile([r['mape_percent'] for r in rr],90)),
                    mape_goal_cells=sum(r['mape_percent']<=6 for r in rr),
                    late_mae_pp=float(np.mean([r['late20_mae_pp'] for r in rr])),stage_pp=float(np.mean([r['stage_error_pp'] for r in rr])),
                    reversal_fraction=float(np.mean([r['acceleration_direction_reversed'] for r in rr])),
                    short_coverage_cells=sum(r['short_coverage'] for r in rr),
                    limitation='Between-cell SD for one seed, not seed uncertainty or a confidence interval.'))
        candidates=[]
        for n in cfg['origins']:
            scopes=[r for r in summary if r['arm']==arm and r['origin']==n and r['horizon'] in ['operational','common']]
            rule=cfg['length_rule']
            if len(scopes)==2 and all(r['mape']<=rule['mape'] and r['late_mae_pp']<=rule['late_mae_pp'] and r['stage_pp']<=rule['stage_pp'] and r['reversal_fraction']<=rule['reversal_fraction'] for r in scopes):candidates.append(n)
        chosen=None
        for n in candidates:
            close=True
            for h in ['operational','common']:
                rr=[r for r in summary if r['arm']==arm and r['horizon']==h];r=next(r for r in rr if r['origin']==n)
                close &= all(r[key]<=min(a[key] for a in rr)+tol for key,tol in [('mape',rule['near_best_mape_pp']),('late_mae_pp',rule['near_best_late_pp']),('stage_pp',rule['near_best_stage_pp'])])
            if close:chosen=n;break
        recommendations.append(dict(arm=arm,exploratory_qualifying_lengths=candidates,provisional_recommendation=chosen,
            final_recommendation=None,reason='Five-seed stability and complete shape/knee diagnostics still required.' if chosen else 'No length meets unchanged exploratory bands and near-best conditions.',
            caveat='Matched recommendation serves only the early-observable subset; cannot extend to all development cells.'))
    save_json(root/'length_summary.json',dict(dataset=cfg['dataset'],summary=summary,recommendations=recommendations,
        verification_sha256=sha(root/'verification.json'),goal_complete=False,independent_test=False))
    print(json.dumps(recommendations),flush=True)
    for r in summary:
        if r['horizon']=='common':print(json.dumps(r),flush=True)


if __name__=='__main__':main()
