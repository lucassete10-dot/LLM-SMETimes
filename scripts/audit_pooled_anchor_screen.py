"""Frozen inner-score candidate selection, never outer data."""
import json
from pathlib import Path
import numpy as np
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    root=Path('experiments/pooled_anchor/20261005T090310_249870Z');cfg=json.loads((root/'config.json').read_text());assert json.loads((root/'independent_verification.json').read_text())['passed']
    old=Path('experiments/nasa_fade_rate_fourfold/20261005T085132_456474Z');ocfg=json.loads((old/'config.json').read_text());assert cfg['plans']==ocfg['plans'] and json.loads((old/'verification.json').read_text())['passed']
    rows=json.loads((root/'summary.json').read_text());refs=json.loads((old/'summary.json').read_text());selected=[];reference=[];control=[];fields=('score','mape_percent','late_mape','endpoint_pp','drop_error_pp')
    for plan in cfg['plans']:
        new=min([r for r in rows if r['fold']==plan['name'] and r['candidate']!=0],key=lambda r:r['score'])
        ref=next(r for r in refs if r['fold']==plan['name'] and r['model']=='strong_ridge');selected.append(new);reference.append(ref);control.append(next(r for r in rows if r['fold']==plan['name'] and r['candidate']==0))
        save_json(root/plan['name']/'selection.json',dict(selected=new,used_only_validation=True,outer_loaded=False))
    means={name:{f:float(np.mean([r[f] for r in rr])) for f in fields} for name,rr in [('new',selected),('strong_ridge',reference),('fixed_rate',control)]}
    wins=sum(n['score']<o['score'] for n,o in zip(selected,reference));passed=wins>=3 and means['new']['score']<means['strong_ridge']['score'] and all(means['new'][f]<=means['strong_ridge'][f] for f in ('endpoint_pp','drop_error_pp'))
    output=dict(means=means,better_folds=wins,folds=4,selected=selected,expand_formal=bool(passed),source_summary_sha256=sha(root/'summary.json'),reference_config_sha256=sha(old/'config.json'),reference_summary_sha256=sha(old/'summary.json'),source_sha256=sha(__file__),aggregation='one-seed inner validation, selected among four preregistered candidates; not formal outer result',goal_complete=False)
    save_json(root/'screening_decision.json',output);print(json.dumps(output),flush=True)


if __name__=='__main__':main()
