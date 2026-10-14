"""First UConn future-curve smoke: split by cells within known protocol groups."""
from copy import deepcopy
from datetime import datetime,timezone
import json,logging,shutil,time,traceback
from pathlib import Path
import joblib,numpy as np
from battery_smetimes import uconn_sparse_prefix as model
from scripts.train_continuous_length_pilot import evaluate
from scripts.prepare_same_type_forecast import sha,save_json


def main():
    indexpath=Path('data/interim/uconn_causal_cycles/20261008T073156_095768Z/index.json')
    index=json.loads(indexpath.read_text()); records={r['cell_id']:r for r in index['cells']}
    rng=np.random.default_rng(20261008); roles=dict(train=[],validation=[],test=[])
    for group in range(1,12):
        keys=sorted(k for k,r in records.items() if r['group_id']==group)
        rng.shuffle(keys); roles['test'].append(keys[0]); roles['validation'].append(keys[1]); roles['train']+=keys[2:]
    assert len(roles['train'])==42 and len(roles['validation'])==len(roles['test'])==11
    assert not set(roles['train'])&set(roles['validation']+roles['test'])
    cells={}
    for cid in roles['train']+roles['validation']:
        rec=records[cid]; assert sha(Path(rec['path']))==rec['sha256']
        with np.load(rec['path'],allow_pickle=False) as a: cells[cid]={k:a[k].copy() for k in a.files}
        cells[cid]['group_id']=rec['group_id']; cells[cid]['physical_family']=rec['physical_family']
    origins=[10,20,30,50,75,100,150,200]
    root=Path('experiments/uconn_sparse_probe')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True); (root/'source_snapshot').mkdir()
    sources=[Path(__file__),Path('battery_smetimes/uconn_sparse_prefix.py'),Path('scripts/train_continuous_length_pilot.py')]
    for p in sources: shutil.copy2(p,root/'source_snapshot'/p.name)
    save_json(root/'config.json',dict(**roles,origins=origins,seeds=[20261008],candidates=['pulse','group'],
        index_path=str(indexpath),index_sha256=sha(indexpath),source_sha256={str(p):sha(p) for p in sources},
        label_reference_ah=1.2,label_reference_status='Provisional typical manufacturer value; exact experimental nominal version not certified.',
        measured_current_temperature_available=False,input_contract='Visible RPT pulse voltage and known experimental group ID only; no capacity/SOH/coulomb-SOC input.',
        common_cutoff=200,independent_test=False,outer_loaded=False,goal_complete=False,
        limitation='Exploratory one split/seed; group IDs are known protocols without verified physical protocol mapping; missing interval cutoffs retained.'))
    logger=logging.getLogger('uconn_sparse_probe'); logger.setLevel(logging.INFO)
    handler=logging.FileHandler(root/'training.log',encoding='utf-8'); logger.addHandler(handler)
    rows=[]; done=0; start=time.perf_counter(); print('RUN='+str(root.resolve()),flush=True)
    try:
        for origin in origins:
            targets={}
            for cid,cell in cells.items():
                mask=cell['source_cycle']>origin
                targets[cid]=(cell['source_cycle'][mask],cell['capacity_ah'][mask]/1.2)
            for block in ('pulse','group'):
                bundle=model.fit(cells,roles['train'],targets,origin,20261008,block)
                folder=root/str(origin)/block; folder.mkdir(parents=True); path=folder/'model.joblib'
                joblib.dump(bundle,path); replay=joblib.load(path); arrays={}; batch=[]
                for cid in roles['validation']:
                    c,y=targets[cid]; p=model.predict(bundle,cells[cid],c)
                    np.testing.assert_allclose(p,model.predict(replay,cells[cid],c),rtol=0,atol=1e-12)
                    poison=deepcopy(cells[cid]); poison['pulse_voltage'][poison['source_cycle']>origin]=np.inf
                    poison['capacity_ah']=np.inf; poison['soh_initial']=np.inf; poison['rpt']=np.inf
                    np.testing.assert_allclose(p,model.predict(bundle,poison,c),rtol=0,atol=1e-12)
                    arrays.update({cid+'__'+k:v for k,v in [('cycle',c),('truth',y),('prediction',p)]})
                    for horizon,mask in [('operational',np.ones(len(c),bool)),('common',c>200)]:
                        rec=dict(cell_id=cid,origin=origin,candidate=block,seed=20261008,horizon=horizon,
                            actual_input_rpts=int(np.sum(cells[cid]['source_cycle']<=origin)),**evaluate(c[mask],y[mask],p[mask]))
                        rows.append(rec); batch.append(rec)
                    for distance in (50,100,200):
                        mask=c<=origin+distance
                        if mask.sum()>=2:
                            rec=dict(cell_id=cid,origin=origin,candidate=block,seed=20261008,horizon=f'fixed_{distance}',
                                full_horizon_covered=bool(c[-1]>=origin+distance),**evaluate(c[mask],y[mask],p[mask]))
                            rows.append(rec); batch.append(rec)
                save_json(folder/'metrics.json',batch); np.savez_compressed(folder/'predictions.npz',**arrays)
                save_json(folder/'config.json',dict(model_sha256=sha(path),train=roles['train'],validation=roles['validation']))
                done+=1; save_json(root/'all_rows.json',rows); save_json(root/'progress.json',dict(completed_fits=done,planned_fits=16,terminal=False))
                s=[r for r in batch if r['horizon']=='common']; message=f'N={origin} {block}: common MAPE={np.mean([r["mape_percent"] for r in s]):.3f}, late MAEpp={np.mean([r["late20_mae_pp"] for r in s]):.3f}'
                logger.info(message); print(message,flush=True)
        save_json(root/'progress.json',dict(completed_fits=done,terminal=True,success=True,seconds=time.perf_counter()-start,goal_complete=False))
    except BaseException:
        logger.exception('FAILED'); save_json(root/'failure.json',dict(traceback=traceback.format_exc()))
        save_json(root/'progress.json',dict(completed_fits=done,terminal=True,success=False)); raise
    finally:
        handler.close(); logger.removeHandler(handler)


if __name__=='__main__':main()
