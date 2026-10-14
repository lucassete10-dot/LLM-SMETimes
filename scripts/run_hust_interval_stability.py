"""Paired five-seed stability at fixed N=100 on the first development fold."""
from datetime import datetime,timezone
import json,subprocess,sys,traceback
from pathlib import Path
import numpy as np
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    root=Path('experiments/hust_interval_stability')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True)
    index='data/interim/hust_voltage_intervals/20261008T074246_799565Z/index.json'
    reference=Path('experiments/hust_voltage_intervals/20261008T074922_391629Z')
    assert json.loads((reference/'verification.json').read_text())['passed']
    reference_cfg=json.loads((reference/'config.json').read_text())
    cfg=dict(physics_index=index,index_sha256=sha(Path(index)),fold=1,origin=100,
        seeds=list(range(20261008,20261013)),first_seed_run=str(reference.resolve()),
        train=reference_cfg['train'],validation=reference_cfg['validation'],test=reference_cfg['test'],
        independent_test=False,goal_complete=False,
        limitation='Five seeds on one whole-cell development fold; not five seeds on every fold, not independent confirmation.')
    save_json(root/'config.json',cfg)
    runs=[dict(seed=20261008,run=str(reference.resolve()))]
    print('STABILITY='+str(root.resolve()),flush=True)
    try:
        for seed in cfg['seeds'][1:]:
            cmd=[sys.executable,'-X','utf8','-m','scripts.train_hust_length_pair','--physics-index',index,
                 '--cv-fold','1','--origins','100','--seed',str(seed)]
            run=None
            with (root/f'seed_{seed}.log').open('w',encoding='utf-8') as log:
                p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
                for line in p.stdout:
                    log.write(line);log.flush();print('SEED',seed,line.rstrip(),flush=True)
                    if line.startswith('RUN='):run=Path(line.strip()[4:])
                if p.wait()!=0:raise RuntimeError(f'seed {seed} failed')
            assert run is not None
            proof=subprocess.run([sys.executable,'-X','utf8','-m','scripts.verify_hust_length_pair',str(run)],capture_output=True,text=True,encoding='utf-8')
            (root/f'seed_{seed}_verification.log').write_text(proof.stdout+proof.stderr,encoding='utf-8')
            if proof.returncode:raise RuntimeError(f'seed {seed} verification failed')
            runs.append(dict(seed=seed,run=str(run)));save_json(root/'runs.json',runs)
        rows=[]
        for rec in runs:
            run=Path(rec['run']);local=json.loads((run/'config.json').read_text())
            for role in ['train','validation','test']:assert local[role]==cfg[role]
            rows.extend(r for r in json.loads((run/'all_rows.json').read_text()) if r['origin']==100)
        summary=[]
        for horizon in ['operational','common']:
            seed_means=[]
            for seed in cfg['seeds']:
                means={}
                for candidate in ['point','voltage_intervals']:
                    rr=[r for r in rows if r['horizon']==horizon and r['seed']==seed and r['candidate']==candidate]
                    assert len(rr)==len(cfg['validation'])
                    means[candidate]={m:float(np.mean([r[m] for r in rr])) for m in ['mape_percent','late20_mae_pp','stage_error_pp','acceleration_direction_reversed']}
                seed_means.append(dict(seed=seed,**means))
            for candidate in ['point','voltage_intervals']:
                summary.append(dict(candidate=candidate,horizon=horizon,cells=len(cfg['validation']),seeds=5,
                    means={m:float(np.mean([r[candidate][m] for r in seed_means])) for m in seed_means[0][candidate]},
                    seed_sd={m:float(np.std([r[candidate][m] for r in seed_means],ddof=1)) for m in seed_means[0][candidate]}))
            save_json(root/f'{horizon}_seed_means.json',seed_means)
        save_json(root/'all_rows.json',rows);save_json(root/'summary.json',summary)
        save_json(root/'progress.json',dict(terminal=True,success=True,seeds_completed=5,goal_complete=False))
        print(json.dumps(summary),flush=True)
    except BaseException:
        save_json(root/'failure.json',dict(traceback=traceback.format_exc(),runs=runs))
        save_json(root/'progress.json',dict(terminal=True,success=False,seeds_completed=len(runs)))
        raise


if __name__=='__main__':main()
