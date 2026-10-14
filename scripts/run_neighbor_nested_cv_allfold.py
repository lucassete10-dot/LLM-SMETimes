"""Sequential five-parent training-only runs plus independent verification."""
import json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scripts.prepare_same_type_forecast import save_json,sha

def main():
    screen=Path('experiments/neighbor_nested_cv/20261005T143522_283877Z');assert json.loads((screen/'independent_verification.json').read_text())['screen_passed']
    root=Path('experiments/neighbor_nested_cv_allfold')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True)
    save_json(root/'config.json',dict(parent_folds=list(range(5)),seed=20261005,preregistration_sha256=sha('docs/soh_neighbor_nested_cv_allfold_20261005.md'),trainer_sha256=sha('scripts/train_neighbor_nested_cv_fold.py'),verifier_sha256=sha('scripts/verify_neighbor_nested_cv.py'),orchestrator_sha256=sha(__file__),outer_loaded=False));records=[]
    for index in range(5):
        result=subprocess.run([sys.executable,'-X','utf8','-m','scripts.train_neighbor_nested_cv_fold','--parent-fold',str(index)],capture_output=True,text=True,encoding='utf-8');save_json(root/f'fold{index+1}_training_process.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr));print(result.stdout,flush=True)
        if result.returncode:raise RuntimeError('trainer failed: '+result.stderr)
        paths=[line[4:] for line in result.stdout.splitlines() if line.startswith('RUN=')];assert len(paths)==1;run=Path(paths[0]);verification=subprocess.run([sys.executable,'-X','utf8','-m','scripts.verify_neighbor_nested_cv','--run',str(run)],capture_output=True,text=True,encoding='utf-8');save_json(root/f'fold{index+1}_verification_process.json',dict(returncode=verification.returncode,stdout=verification.stdout,stderr=verification.stderr));print(verification.stdout,flush=True)
        if verification.returncode:raise RuntimeError('verifier failed: '+verification.stderr)
        assert json.loads((run/'independent_verification.json').read_text())['passed'];sel=json.loads((run/'selection.json').read_text());records.append(dict(parent_fold=index+1,run=str(run.resolve()),run_config_sha256=sha(run/'config.json'),verification_sha256=sha(run/'independent_verification.json'),selection_sha256=sha(run/'selection.json'),**sel));save_json(root/'progress.json',dict(verified_parent_folds=len(records),complete=False,outer_loaded=False));save_json(root/'fold_results.json',records)
    wins=sum(r['selected']['score']<r['control']['score'] for r in records);means={role:{f:float(np.mean([r[role][f] for r in records])) for f in ('score','mape_percent','endpoint_pp','drop_error_pp')} for role in ('selected','control')};expand=wins>=4 and means['selected']['score']<means['control']['score'] and all(means['selected'][f]<=means['control'][f] for f in ('endpoint_pp','drop_error_pp'))
    save_json(root/'screening_decision.json',dict(parent_score_wins=wins,means=means,expand_formal=expand,outer_loaded=False,goal_complete=False));save_json(root/'progress.json',dict(verified_parent_folds=5,complete=True,outer_loaded=False,goal_complete=False));print(json.dumps(dict(wins=wins,means=means,expand=expand)),flush=True)

if __name__=='__main__':main()
