"""Summarize preregistered screening, not a formal goal acceptance report."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text())
    assert (cfg.get('fivefold_shape',False) or cfg.get('fivefold_rate',False)) and json.loads((root/'independent_verification.json').read_text())['passed']
    comparison=json.loads((root/'current_primary_validation_comparison.json').read_text())
    assert comparison['used_only_train_validation'] and not comparison['outer_loaded']
    fields=('score','mape_percent','late_mape','endpoint_pp','drop_error_pp');new=[];old=[];provenance=[]
    assert len(comparison['reports'])==len(cfg['plans'])==5
    for plan,report in zip(cfg['plans'],comparison['reports']):
        assert plan['name']==report['fold'];selection=report['primary']['selected']
        selection_file=Path(selection['selection_file']);assert sha(selection_file)==selection['selection_sha256']
        folder=selection_file.parent/('candidate_'+str(selection['candidate']))
        assert sha(folder/'model.joblib')==selection['model_sha256']
        rows=json.loads((folder/'validation_metrics.json').read_text());assert {r['cell_id'] for r in rows}==set(plan['validation'])
        old.append(dict(score=selection['score'],**{f:float(np.mean([r[f] for r in rows])) for f in fields[1:]}))
        new.append(report['new']);provenance.append(dict(file=str((folder/'validation_metrics.json').resolve()),sha256=sha(folder/'validation_metrics.json')))
    means={name:{f:float(np.mean([r[f] for r in records])) for f in fields} for name,records in [('new',new),('primary',old)]}
    wins=sum(n['score']<o['score'] for n,o in zip(new,old))
    passed=wins>=4 and means['new']['score']<means['primary']['score'] and all(means['new'][f]<=means['primary'][f] for f in ('endpoint_pp','drop_error_pp'))
    output=dict(means=means,new_better_folds=wins,folds=5,expand_formal=bool(passed),aggregation='equal-cell within fold, then equal-fold; one seed; development validation',source_metrics=provenance,comparison_sha256=sha(root/'current_primary_validation_comparison.json'),outer_loaded=False,goal_complete=False,source_sha256=sha(__file__))
    save_json(root/'screening_decision.json',output);print(json.dumps(output,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
