"""MIT-only paired fixed-cell CV: 300s vs completed past signal features."""
import json,time,traceback,argparse
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import h5py,joblib,numpy as np
from audit_mit_label_free_extraction_v2 import MIT_SOURCE
from prepare_mit import read_ref
from battery_smetimes import continuous_length as model
from battery_smetimes.mit_completed_signal import cycle_features
from scripts.train_core_length_cv import load
from scripts.train_continuous_length_pilot import target,evaluate
from scripts.prepare_same_type_forecast import sha,save_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path)
    ap.add_argument('--origins',type=int,nargs='+',default=[50,100])
    ap.add_argument('--seeds',type=int,nargs='+',default=[20261008]);args=ap.parse_args()
    baseline=Path('experiments/core_length_cv/20261008T080847_529428Z')
    old=json.loads((baseline/'config.json').read_text());plans=old['plans'];cells=load(plans[0])
    root=Path('experiments/mit_completed_signal')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True)
    cfg=dict(plans=plans,origins=args.origins,seeds=args.seeds,baseline=str(baseline),
        arms=['prefix300','completed_past'],independent_test=False,outer_loaded=False,
        input_contract='Completed past 4C discharge V/I/t/T at source cycles <= N. No summary capacity or future input.',
        selection='Development exploration only. No adoption unless tail and stage errors improve without material overall regression; report all cells, not just four examples.',
        source_sha256={str(p):sha(p) for p in [Path(__file__),Path('battery_smetimes/mit_completed_signal.py'),Path('battery_smetimes/continuous_length.py')]},
        cache=str(args.cache) if args.cache else None)
    save_json(root/'config.json',cfg);start=time.perf_counter();rows=[];manifest=[];expanded={}
    try:
        assert not set(cells)&set(plans[0]['test'])
        if args.cache:
            manifest=json.loads((args.cache/'feature_manifest.json').read_text())
            for r in manifest:
                cid=r['cell_id'];p=args.cache/(cid+'.features.npz');assert sha(p)==r['sha256']
                with np.load(p) as d:
                    expanded[cid]=dict(cells[cid],early_cycle=d['cycle'].copy(),early_raw=d['raw'].copy(),conditions=d['conditions'].copy())
        else:
            for batch,source in MIT_SOURCE.items():
                with h5py.File(source,'r') as f:
                    for cid,cell in cells.items():
                        b,idx=map(int,cid[1:].split('c'))
                        if b!=batch:continue
                        st=f['batch'];summary=f[st['summary'][idx,0]];trace=f[st['cycles'][idx,0]]
                        numbers=np.asarray(summary['cycle']).reshape(-1).astype(int)
                        if len(numbers)!=len(trace['I']):raise ValueError('cycle/trace mismatch')
                        cc=[];xx=[];excluded=[]
                        for j,num in enumerate(numbers):
                            if num>100:continue
                            try:
                                vals={k:read_ref(f,trace[k][j,0]) for k in ('t','I','V','T')}
                                x=cycle_features(vals['t'],vals['I'],vals['V'],vals['T'])
                            except ValueError as exc:
                                excluded.append(dict(cycle=int(num),reason=str(exc)));continue
                            cc.append(num);xx.append(x)
                        sc,sr,cond=model.signal_history(cell);pc=np.asarray(cc);pr=np.asarray(xx)
                        common=np.intersect1d(sc,pc)
                        expanded[cid]=dict(cell,early_cycle=common,early_raw=np.c_[sr[np.searchsorted(sc,common)],pr[np.searchsorted(pc,common)]],conditions=cond)
                        path=root/(cid+'.features.npz')
                        np.savez_compressed(path,cycle=common,raw=expanded[cid]['early_raw'],conditions=cond)
                        manifest.append(dict(cell_id=cid,sha256=sha(path),observations=len(common),excluded=excluded))
                        print('FEATURE',cid,len(common),flush=True)
            save_json(root/'feature_manifest.json',manifest)
        assert set(expanded)==set(cells)
        for n in args.origins:
            usable=[]
            for cid in cells:
                try:
                    model.features(cells[cid],n);model.features(expanded[cid],n)
                    if len(target(cells[cid],n)[0])>=2:usable.append(cid)
                except ValueError:pass
            targets={c:target(cells[c],n) for c in usable}
            for plan in plans:
                train=[c for c in plan['train'] if c in usable];val=[c for c in plan['validation'] if c in usable]
                for seed in args.seeds:
                    for arm,src in [('prefix300',cells),('completed_past',expanded)]:
                        folder=root/plan['name']/str(n)/str(seed)/arm;folder.mkdir(parents=True)
                        bundle=model.fit(src,train,n,'point','VIT',1,seed,targets)
                        joblib.dump(bundle,folder/'model.joblib');replay=joblib.load(folder/'model.joblib');arrays={}
                        for cid in val:
                            c,y=targets[cid];p=model.predict(bundle,src[cid],c)
                            np.testing.assert_allclose(p,model.predict(replay,src[cid],c),rtol=0,atol=1e-12)
                            poison=deepcopy(src[cid]);poison['raw'][poison['cycle']>n]=np.inf
                            if 'early_raw' in poison:poison['early_raw'][poison['early_cycle']>n]=np.inf
                            poison['soh']=np.inf;poison['label_cycle']=np.array([np.inf])
                            np.testing.assert_allclose(p,model.predict(bundle,poison,c),rtol=0,atol=1e-12)
                            arrays.update({cid+'__'+k:v for k,v in [('cycle',c),('truth',y),('prediction',p)]})
                            mask=c>100
                            rows.append(dict(cell_id=cid,fold=plan['name'],origin=n,seed=seed,arm=arm,**evaluate(c[mask],y[mask],p[mask])))
                        np.savez_compressed(folder/'predictions.npz',**arrays)
                        save_json(folder/'config.json',dict(train=train,validation=val,model_sha256=sha(folder/'model.joblib'),future_poison_and_reload_passed=True))
                        save_json(root/'all_rows.json',rows)
                        rr=[r for r in rows if r['origin']==n and r['seed']==seed and r['arm']==arm]
                        print('FIT',n,plan['name'],seed,arm,'MAPE',np.mean([r['mape_percent'] for r in rr]),'late',np.mean([r['late20_mae_pp'] for r in rr]),'stage',np.mean([r['stage_error_pp'] for r in rr]),flush=True)
        save_json(root/'progress.json',dict(success=True,terminal=True,seconds=time.perf_counter()-start,goal_complete=False))
    except BaseException:
        save_json(root/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
