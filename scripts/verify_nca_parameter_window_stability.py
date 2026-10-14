"""Replay TRAIN truncation solvers and separately recompute curve differences."""
import json,sys
from pathlib import Path
import numpy as np
from battery_smetimes.curve_fidelity import shape_parameters
from scripts.train_tail_boost import load_cells,NCA
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.verify_nasa_robust_inner import close


def main(root):
    cfg=json.loads((root/'config.json').read_text());plan=cfg['plan'];cells=load_cells(dict(plan,validation=[],test=[]))
    assert set(cells)==set(plan['train']) and not set(cells)&set(plan['validation']+plan['test'])
    assert sha(Path(cfg['parent'])/'config.json')==cfg['parent_config_sha256'] and sha(NCA/'index.json')==cfg['nca_index_sha256']
    for f,d in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/Path(f).name)==d
    rows=json.loads((root/'rows.json').read_text());summary=json.loads((root/'summary.json').read_text());expected=[];fits=39
    assert len(rows)==117 and len({r['cell_id'] for r in rows})==39
    full={k:shape_parameters(c['cycle'],c['truth'],2) for k,c in cells.items()}
    with np.load(root/'reconstructions.npz',allow_pickle=False) as arrays:
        for r in rows:
            k,cutoff=r['cell_id'],r['cutoff'];c,y=cells[k]['cycle'],cells[k]['truth'];take=c<=cutoff
            assert cutoff in cfg['cutoffs'] and r['available']==bool(take.sum()>=2)
            if not r['available']:expected.append(r);continue
            a=shape_parameters(c[take],y[take],2);fits+=1;af=full[k];h=(c-100)/100
            p=a[0]-a[1]*h-a[2]*h*h;pf=af[0]-af[1]*h-af[2]*h*h;after=c>cutoff
            for f,v in [('cycle',c),('truth',y),('short_fit',p),('full_fit',pf)]:np.testing.assert_array_equal(arrays[k+'__'+str(cutoff)+'__'+f],v)
            e=dict(cell_id=k,cutoff=cutoff,available=True,actually_truncated=bool(after.any()),full_coefficients=af.tolist(),short_coefficients=a.tolist(),
                observed_end_cycle=float(c[take][-1]),full_end_cycle=float(c[-1]),full_label_points=len(c),short_label_points=int(take.sum()),
                acceleration_change=float(a[2]-af[2]),end_reconstruction_change_pp=float(100*(p[-1]-pf[-1])),abs_end_reconstruction_change_pp=float(100*abs(p[-1]-pf[-1])),
                fitted_window_mape_percent=float(100*np.average(abs(p[take]-y[take])/y[take])),
                beyond_window_mape_percent=float(100*np.average(abs(p[after]-y[after])/y[after])) if after.any() else None,
                full_fit_beyond_window_mape_percent=float(100*np.average(abs(pf[after]-y[after])/y[after])) if after.any() else None,not_predictive_model=True)
            close(r,e);expected.append(e)
        assert len(arrays.files)==4*sum(r['available'] for r in rows)
    for cutoff in cfg['cutoffs']:
        rr=[r for r in expected if r['cutoff']==cutoff and r['available'] and r['actually_truncated']]
        e=dict(truncated_cells=len(rr),unavailable_cells=sum(r['cutoff']==cutoff and not r['available'] for r in expected),
            **{f:float(np.mean([r[f] for r in rr])) if rr else None for f in ('abs_end_reconstruction_change_pp','fitted_window_mape_percent','beyond_window_mape_percent','full_fit_beyond_window_mape_percent')},
            large_end_change_cells=sum(r['abs_end_reconstruction_change_pp']>5 for r in rr))
        close(summary['groups'][str(cutoff)],e)
    assert summary['shape_fits']==fits==156 and summary['predictive_fits']==0
    save_json(root/'independent_verification.json',dict(passed=True,train_cells=39,rows=117,shape_solver_replays=156,
        scope='Shared constrained solver replay; separate truncation and curve/metric arithmetic, not early-feature forecasting.',
        original_validation_loaded=False,outer_loaded=False,verifier_sha256=sha(Path(__file__)),goal_complete=False))
    print('PASS:156TRAIN shape replays/117window rows, no predictive fitting or original validation/test reads.')


if __name__=='__main__':main(Path(sys.argv[1]))
