"""Render real completed v3 TRAIN/VAL curves without requesting TEST."""
import argparse,json,subprocess,sys
from pathlib import Path
from scripts.run_llm_curve_v3_r0 import read
from scripts.audit_llm_curve_repair import load_frozen,fingerprint
from scripts.prepare_same_type_forecast import save_json
from battery_smetimes.llm_v3_workflow import SEEDS


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--roles',nargs='+',choices=['training','validation'],default=['validation'])
    ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args()
    run=args.run;cfg=read(run/'config.json');proof=read(run/'repair_verification.json')
    if cfg.get('selection_version')!='curve_quality_v3':raise ValueError('trial report requires actual v3 model')
    if not proof['passed'] or proof['model_sha256']!=fingerprint(run/'model.pt'):
        raise ValueError('trial replay/checkpoint evidence invalid')
    if not set(args.roles)<=set(proof['role_scope']):raise ValueError('requested report role was not audited')
    _,_,plan,_,_=load_frozen(cfg);task={'dataset':cfg['dataset'],'fold':cfg['fold'],'seed':cfg['seed'],
        'state':'complete','run':str(run.resolve()),'model_sha256':proof['model_sha256'],
        'role_counts':{r:len(plan[r]) for r in ('train','validation','test')}}
    bridge=args.out/'source_bridge';bridge.mkdir(parents=True,exist_ok=True)
    save_json(bridge/'queue.json',{'tasks':[task],'seeds':list(SEEDS),'origin':cfg['origin'],
        'candidate_test_not_requested':True,'goal_complete':False})
    if args.prepare_only:print(json.dumps({'bridge':str(bridge),'roles':args.roles}));return
    for role in args.roles:
        subprocess.run([sys.executable,'-m','scripts.report_llm_full_dataset','--queue',str(bridge),
            '--dataset',cfg['dataset'],'--progress-only','--figure-roles',role,
            '--available-roles',*args.roles,'--quality-v3','--output-dir',str(args.out/role)],check=True)
    save_json(args.out/'trial_report_manifest.json',{'run':str(run.resolve()),'roles':args.roles,
        'model_sha256':proof['model_sha256'],'selection_version':cfg['selection_version'],
        'selected_epoch':read(run/'result.json')['selected_epoch'],'candidate_test_not_requested':True,
        'visual_review':'pending','goal_complete':False})


if __name__=='__main__':main()
