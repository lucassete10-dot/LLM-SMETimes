"""Independent five-parent expansion aggregation from verified child runs."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.prepare_same_type_forecast import sha,save_json

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text());assert json.loads((root/'progress.json').read_text())['complete']
    for file,key in [('docs/soh_neighbor_nested_cv_allfold_20261005.md','preregistration_sha256'),('scripts/train_neighbor_nested_cv_fold.py','trainer_sha256'),('scripts/verify_neighbor_nested_cv.py','verifier_sha256'),('scripts/run_neighbor_nested_cv_allfold.py','orchestrator_sha256')]:assert sha(file)==cfg[key]
    records=json.loads((root/'fold_results.json').read_text());assert [r['parent_fold'] for r in records]==[1,2,3,4,5];values={role:[] for role in ('selected','control')};curves=0;refits=0;wins=0
    for r in records:
        child=Path(r['run']);ccfg=json.loads((child/'config.json').read_text());proof=json.loads((child/'independent_verification.json').read_text())
        assert ccfg['parent_plan']['name']=='NCA_fold'+str(r['parent_fold']) and proof['passed'] and not proof['original_validation_loaded'] and not proof['outer_loaded'];curves+=proof['curves'];refits+=proof['independent_hgb_refits']
        for file,key in [('config.json','run_config_sha256'),('independent_verification.json','verification_sha256'),('selection.json','selection_sha256')]:assert sha(child/file)==r[key]
        selection=json.loads((child/'selection.json').read_text());assert all(selection[f]==r[f] for f in selection)
        # Recompute candidate scores and cell-equal means from independently replayed OOF records.
        candidates=[]
        for index in range(7):
            rows=json.loads((child/f'pooled_candidate_{index}_metrics.json').read_text());assert len(rows)==len({a['cell_id'] for a in rows})==len(ccfg['parent_plan']['train'])
            assert {a['cell_id'] for a in rows}==set(ccfg['parent_plan']['train']);v=np.array([(a['mape_percent']+a['late_mape']+a['endpoint_pp']+a['drop_error_pp'])/4 for a in rows]);candidates.append(dict(candidate=index,score=float(v.mean()+.25*np.quantile(v,.9)),**{f:float(np.mean([a[f] for a in rows])) for f in ('mape_percent','endpoint_pp','drop_error_pp')}))
        selected=min(candidates,key=lambda a:(a['score'],a['candidate']));control=candidates[0]
        for role,value in [('selected',selected),('control',control)]:
            for f,v in value.items():np.testing.assert_allclose(v,r[role][f],rtol=0,atol=1e-10)
            values[role].append(value)
        wins+=selected['score']<control['score']
    means={role:{f:float(np.mean([v[f] for v in rows])) for f in ('score','mape_percent','endpoint_pp','drop_error_pp')} for role,rows in values.items()};expand=wins>=4 and means['selected']['score']<means['control']['score'] and all(means['selected'][f]<=means['control'][f] for f in ('endpoint_pp','drop_error_pp'));stored=json.loads((root/'screening_decision.json').read_text());assert stored['parent_score_wins']==wins and stored['expand_formal']==expand
    for role in means:
        for field,v in means[role].items():np.testing.assert_allclose(v,stored['means'][role][field],rtol=0,atol=1e-10)
    save_json(root/'independent_verification.json',dict(passed=True,curves=curves,independent_hgb_refits=refits,parent_score_wins=int(wins),means=means,expand_formal=bool(expand),source_sha256=sha(__file__),outer_loaded=False,goal_complete=False));print(json.dumps(dict(passed=True,curves=curves,wins=int(wins),expand=bool(expand))),flush=True)

if __name__=='__main__':main()
