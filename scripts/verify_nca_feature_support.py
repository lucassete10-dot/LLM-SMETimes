"""Separate descriptive-support arithmetic from forecast fitting."""
import json,sys
from pathlib import Path
import numpy as np
from scripts.train_tail_boost import load_cells
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.verify_nasa_robust_inner import close


def main(root):
    cfg=json.loads((root/'config.json').read_text());rows=json.loads((root/'oof_rows.json').read_text());expanded=json.loads((root/'support_rows.json').read_text())
    report=json.loads((root/'support_diagnosis.json').read_text());cells=load_cells(dict(cfg['plan'],validation=[],test=[]))
    assert set(cells)==set(cfg['plan']['train']) and len(rows)==len(expanded)==780
    for f,d in report['input_sha256'].items():assert sha(root/f)==d
    for r,out in zip(rows,expanded):
        for f,v in r.items():assert out[f]==v
        tr=next(s['train'] for s in cfg['splits'] if s['fold']==r['diagnostic_fold']);end=float(cells[r['cell_id']]['cycle'][-1])
        last=[float(cells[k]['cycle'][-1]) for k in tr];same=[k for k in tr if tuple(cells[k]['conditions'])==tuple(cells[r['cell_id']]['conditions'])]
        check=dict(requested_end_cycle=end,training_max_observed_cycle=max(last),beyond_training_max=end>max(last),
            support_at_end_fraction=sum(c>=end for c in last)/len(last),same_protocol_members=len(same),
            same_protocol_support_at_end_fraction=sum(float(cells[k]['cycle'][-1])>=end for k in same)/len(same) if same else None,
            same_protocol_context_only=True,not_actual_route=True)
        close({f:out[f] for f in check},check)
    for arm,groups in report['groups'].items():
        rr=[r for r in expanded if r['arm']==arm]
        for name,group in groups.items():
            if name=='all':part=rr
            elif name=='beyond_train_max':part=[r for r in rr if r['beyond_training_max']]
            elif name=='within_train_max':part=[r for r in rr if not r['beyond_training_max']]
            elif name=='low_same_protocol_support':part=[r for r in rr if r['same_protocol_support_at_end_fraction'] is None or r['same_protocol_support_at_end_fraction']<.25]
            else:
                assert name=='other';part=[r for r in rr if r['same_protocol_support_at_end_fraction'] is not None and r['same_protocol_support_at_end_fraction']>=.25]
            close(group,dict(overlapping_rows=len(part),unique_cells=len({r['cell_id'] for r in part}),
                **{f:float(np.mean([r[f] for r in part])) if part else None for f in ('mape_percent','endpoint_pp','drop_error_pp')}))
    for r in report['paired_seed_changes']:
        lookup={(a['arm'],a['cell_id']):a for a in rows if a['seed']==r['seed']}
        delta=[lookup[r['arm'],k]['mape_percent']-lookup['conditions',k]['mape_percent'] for k in cfg['plan']['train']]
        np.testing.assert_allclose(r['mean_paired_mape_change'],np.mean(delta),atol=1e-10,rtol=0)
        np.testing.assert_allclose(r['cell_win_fraction'],sum(v<0 for v in delta)/39,atol=1e-10,rtol=0)
    save_json(root/'support_independent_verification.json',dict(passed=True,rows=780,groups=20,paired_seed_changes=15,
        outer_loaded=False,new_fits=0,scope='Descriptive support arithmetic only, not route or causal inference.',verifier_sha256=sha(Path(__file__))))
    print('PASS:780supportrows/20groups/15pairedseed changes; no new fitting or filtering.')


if __name__=='__main__':main(Path(sys.argv[1]))
