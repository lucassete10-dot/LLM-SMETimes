"""Descriptive OOF support audit; no fit or model selection."""
import json,sys
from pathlib import Path
import numpy as np
from scripts.train_tail_boost import load_cells
from scripts.prepare_same_type_forecast import sha,save_json


def main(root):
    cfg=json.loads((root/'config.json').read_text());cells=load_cells(dict(cfg['plan'],validation=[],test=[]))
    rows=json.loads((root/'oof_rows.json').read_text());expanded=[]
    for r in rows:
        split=next(s for s in cfg['splits'] if s['fold']==r['diagnostic_fold']);k=r['cell_id'];end=float(cells[k]['cycle'][-1])
        ends=np.array([cells[t]['cycle'][-1] for t in split['train']],float)
        same=[t for t in split['train'] if np.array_equal(cells[t]['conditions'],cells[k]['conditions'])]
        se=np.array([cells[t]['cycle'][-1] for t in same],float)
        expanded.append(dict(**r,requested_end_cycle=end,training_max_observed_cycle=float(ends.max()),
            beyond_training_max=bool(end>ends.max()),support_at_end_fraction=float(np.mean(ends>=end)),
            same_protocol_members=len(same),same_protocol_support_at_end_fraction=float(np.mean(se>=end)) if len(se) else None,
            same_protocol_context_only=True,not_actual_route=True))
    summaries={}
    for arm in cfg['arms']:
        rr=[r for r in expanded if r['arm']==arm];groups={}
        for name,part in [('all',rr),('beyond_train_max',[r for r in rr if r['beyond_training_max']]),('within_train_max',[r for r in rr if not r['beyond_training_max']]),
            ('low_same_protocol_support',[r for r in rr if r['same_protocol_support_at_end_fraction'] is None or r['same_protocol_support_at_end_fraction']<.25]),
            ('other',[r for r in rr if r['same_protocol_support_at_end_fraction'] is not None and r['same_protocol_support_at_end_fraction']>=.25])]:
            groups[name]=dict(overlapping_rows=len(part),unique_cells=len({r['cell_id'] for r in part}),
                **{f:float(np.mean([r[f] for r in part])) if part else None for f in ('mape_percent','endpoint_pp','drop_error_pp')})
        summaries[arm]=groups
    pairs=[]
    for seed in cfg['seeds']:
        for arm in ('prefix','segments','prefix_shuffled'):
            differences=[]
            for k in cfg['plan']['train']:
                a=next(r for r in rows if r['seed']==seed and r['cell_id']==k and r['arm']==arm)
                b=next(r for r in rows if r['seed']==seed and r['cell_id']==k and r['arm']=='conditions')
                differences.append(a['mape_percent']-b['mape_percent'])
            pairs.append(dict(seed=seed,arm=arm,mean_paired_mape_change=float(np.mean(differences)),cell_win_fraction=float(np.mean(np.array(differences)<0))))
    save_json(root/'support_rows.json',expanded);save_json(root/'support_diagnosis.json',dict(groups=summaries,paired_seed_changes=pairs,
        input_sha256={f:sha(root/f) for f in ('config.json','oof_rows.json')},analyzer_sha256=sha(Path(__file__)),
        warning='Descriptive support association, not causal proof, model route, or new filtering rule.',new_fits=0,outer_loaded=False))
    print(json.dumps(dict(prefix=summaries['prefix'],pairs=pairs),ensure_ascii=False),flush=True)


if __name__=='__main__':main(Path(sys.argv[1]))
