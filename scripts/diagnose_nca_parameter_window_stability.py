"""TRAIN-only truncation intervention on coefficient supervision, not forecasting."""
import json,shutil,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from battery_smetimes.curve_fidelity import shape_parameters
from scripts.train_tail_boost import load_cells,NCA
from scripts.prepare_same_type_forecast import sha,save_json

PARENT=Path('experiments/nca_feature_discrimination/20261007T051231_597724Z')
CUTOFFS=[200,400,600]


def main():
    cfg=json.loads((PARENT/'config.json').read_text());plan=cfg['plan'];cells=load_cells(dict(plan,validation=[],test=[]))
    assert set(cells)==set(plan['train']) and not set(cells)&set(plan['validation']+plan['test'])
    root=Path('experiments/nca_parameter_window_stability')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True)
    snap=root/'source_snapshot';snap.mkdir();files=[Path(__file__),Path('battery_smetimes/curve_fidelity.py')]
    for f in files:shutil.copy2(f,snap/f.name)
    save_json(root/'config.json',dict(parent=str(PARENT.resolve()),parent_config_sha256=sha(PARENT/'config.json'),plan=plan,cutoffs=CUTOFFS,
        source_sha256={str(f):sha(f) for f in files},nca_index_sha256=sha(NCA/'index.json'),prepared_sha256=cfg['prepared_sha256'],
        original_validation_loaded=False,outer_loaded=False,predictive_fits=0,scope='Own TRAIN labels with fixed source-cycle truncation; not model forecast accuracy'))
    rows=[];arrays={};fits=0;started=time.perf_counter()
    for k,cell in cells.items():
        c,y=cell['cycle'],cell['truth'];h=(c-100)/100;full=shape_parameters(c,y,2);fits+=1
        pf=full[0]-full[1]*h-full[2]*h*h
        for cutoff in CUTOFFS:
            take=c<=cutoff
            if take.sum()<2:
                rows.append(dict(cell_id=k,cutoff=cutoff,available=False,reason='fewer than two TRAIN labels'));continue
            a=shape_parameters(c[take],y[take],2);fits+=1;p=a[0]-a[1]*h-a[2]*h*h
            after=c>cutoff;delta=100*(p[-1]-pf[-1])
            rows.append(dict(cell_id=k,cutoff=cutoff,available=True,actually_truncated=bool(after.any()),full_coefficients=full.tolist(),short_coefficients=a.tolist(),
                observed_end_cycle=float(c[take][-1]),full_end_cycle=float(c[-1]),full_label_points=len(c),short_label_points=int(take.sum()),
                acceleration_change=float(a[2]-full[2]),end_reconstruction_change_pp=float(delta),abs_end_reconstruction_change_pp=abs(float(delta)),
                fitted_window_mape_percent=float(100*np.mean(abs(p[take]-y[take])/y[take])),
                beyond_window_mape_percent=float(100*np.mean(abs(p[after]-y[after])/y[after])) if after.any() else None,
                full_fit_beyond_window_mape_percent=float(100*np.mean(abs(pf[after]-y[after])/y[after])) if after.any() else None,
                not_predictive_model=True))
            arrays.update({k+'__'+str(cutoff)+'__'+f:v for f,v in [('cycle',c),('truth',y),('short_fit',p),('full_fit',pf)]})
    groups={}
    for cutoff in CUTOFFS:
        rr=[r for r in rows if r['cutoff']==cutoff and r['available'] and r['actually_truncated']]
        groups[str(cutoff)]=dict(truncated_cells=len(rr),unavailable_cells=sum(r['cutoff']==cutoff and not r['available'] for r in rows),
            **{f:float(np.mean([r[f] for r in rr])) if rr else None for f in ('abs_end_reconstruction_change_pp','fitted_window_mape_percent','beyond_window_mape_percent','full_fit_beyond_window_mape_percent')},
            large_end_change_cells=sum(r['abs_end_reconstruction_change_pp']>5 for r in rr))
    save_json(root/'rows.json',rows);np.savez_compressed(root/'reconstructions.npz',**arrays)
    save_json(root/'summary.json',dict(groups=groups,rows=len(rows),train_cells=39,shape_fits=fits,predictive_fits=0,seconds=time.perf_counter()-started,
        warning='Parameter stability on TRAIN labels only. Not early-input forecast accuracy or proof of cause for retained models.',goal_complete=False))
    print(json.dumps(groups),flush=True)


if __name__=='__main__':main()
