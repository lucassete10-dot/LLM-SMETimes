"""Five whole-cell folds with fixed signal windows; preserve each run and failure."""
from datetime import datetime,timezone
import json,subprocess,sys,traceback
from pathlib import Path
import numpy as np
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    root=Path('experiments/hust_voltage_intervals_batches')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True)
    index='data/interim/hust_voltage_intervals/20261008T074246_799565Z/index.json'
    cfg=dict(physics_index=index,index_sha256=sha(Path(index)),folds=[1,2,3,4,5],
        origins=[10,20,30,50,75,100,150,200],seed=20261008,independent_test=False,
        selection_scope='Development CV; five-seed confirmation and knee/path acceptance still required.',
        paired_candidates=['point','voltage_intervals'],goal_complete=False)
    save_json(root/'config.json',cfg)
    completed=[]
    print('BATCH='+str(root.resolve()),flush=True)
    try:
        for fold in cfg['folds']:
            cmd=[sys.executable,'-X','utf8','-m','scripts.train_hust_length_pair','--physics-index',index,'--cv-fold',str(fold)]
            with (root/f'fold_{fold}.log').open('w',encoding='utf-8') as log:
                p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
                run=None
                for line in p.stdout:
                    log.write(line);log.flush();print('FOLD',fold,line.rstrip(),flush=True)
                    if line.startswith('RUN='):run=Path(line.strip()[4:])
                if p.wait()!=0:raise RuntimeError(f'fold {fold} failed; see retained log')
            assert run is not None
            verification=subprocess.run([sys.executable,'-X','utf8','-m','scripts.verify_hust_length_pair',str(run)],capture_output=True,text=True,encoding='utf-8')
            (root/f'fold_{fold}_verification.log').write_text(verification.stdout+verification.stderr,encoding='utf-8')
            if verification.returncode:raise RuntimeError(f'fold {fold} verification failed')
            completed.append(dict(fold=fold,run=str(run),verification_sha256=sha(run/'verification.json')))
            save_json(root/'runs.json',completed)
        rows=[];seen={}
        for rec in completed:
            run=Path(rec['run']);c=json.loads((run/'config.json').read_text())
            assert c['cv_fold']==rec['fold'] and c['test']==json.loads((Path(completed[0]['run'])/'config.json').read_text())['test']
            for cid in c['validation']:
                assert cid not in seen;seen[cid]=rec['fold']
            rows.extend(json.loads((run/'all_rows.json').read_text()))
        assert len(seen)==61
        summary=[]
        for n in cfg['origins']:
            for name in cfg['paired_candidates']:
                for horizon in ['operational','common','fixed_50','fixed_100','fixed_200']:
                    rr=[r for r in rows if r['origin']==n and r['candidate']==name and r['horizon']==horizon]
                    assert len(rr)==61 and len({r['cell_id'] for r in rr})==61
                    summary.append(dict(origin=n,candidate=name,horizon=horizon,cells=len(rr),
                        mape_percent=float(np.mean([r['mape_percent'] for r in rr])),
                        late20_mae_pp=float(np.mean([r['late20_mae_pp'] for r in rr])),
                        stage_error_pp=float(np.mean([r['stage_error_pp'] for r in rr])),
                        reversal_fraction=float(np.mean([r['acceleration_direction_reversed'] for r in rr])),
                        mape_cell_sd=float(np.std([r['mape_percent'] for r in rr],ddof=1))))
        save_json(root/'all_rows.json',rows);save_json(root/'summary.json',summary)
        save_json(root/'progress.json',dict(terminal=True,success=True,completed_folds=5,goal_complete=False))
        for row in summary:
            if row['horizon']=='common':print(json.dumps(row),flush=True)
    except BaseException:
        save_json(root/'failure.json',dict(traceback=traceback.format_exc(),completed=completed))
        save_json(root/'progress.json',dict(terminal=True,success=False,completed_folds=len(completed)))
        raise


if __name__=='__main__':main()
