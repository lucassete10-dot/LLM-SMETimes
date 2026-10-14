"""Independent signed-error and late-slope replay of frozen inner diagnostics."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.prepare_same_type_forecast import sha,save_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    root=ap.parse_args().run;cfg=json.loads((root/'config.json').read_text())
    assert not cfg['outer_predictions_read'] and cfg['new_fits']==0
    parent=Path(cfg['parent']);pc=json.loads((parent/'config.json').read_text())
    assert sha(parent/'config.json')==cfg['parent_config_sha256']
    assert sha(parent/'independent_verification.json')==cfg['parent_proof_sha256']
    for f,h in {**cfg['source_sha256'],**cfg['input_sha256']}.items():assert sha(f)==h
    rows=json.loads((root/'rows.json').read_text());summary=json.loads((root/'summary.json').read_text())
    expected={(p['name'],s,k,a) for p in pc['plans'] for s in pc['seeds'] for k in p['validation'] for a in ('selected','control')}
    assert len(rows)==len(expected)==1240
    assert {(r['fold'],r['seed'],r['cell_id'],r['arm']) for r in rows}==expected
    for r in rows:
        plan=next(p for p in pc['plans'] if p['name']==r['fold'])
        assert r['cell_id'] not in plan['test']
        child=parent/r['fold']/str(r['seed']);sel=json.loads((child/'selection.json').read_text())[r['arm']]
        assert r['candidate']==sel['candidate'] and r['new_weight']==sel['new_weight']
        prefix=r['cell_id']+'__100__'
        with np.load(child/f"candidate_{r['candidate']}"/'validation_predictions.npz') as a:
            c,y,p=[a[prefix+f] for f in ('cycle','truth','prediction')]
        start=(p[0]-y[0])*100;end=(p[-1]-y[-1])*100
        td=(y[0]-y[-1])*100;pd=(p[0]-p[-1])*100;change=td-pd
        mask=c>=c[0]+.75*(c[-1]-c[0])
        if mask.sum()<2:mask[-2:]=True
        t=c[mask]-c[mask].mean()
        true_rate=-np.sum(t*y[mask])/np.sum(t*t)*10000
        pred_rate=-np.sum(t*p[mask])/np.sum(t*t)*10000
        values=dict(start_signed_pp=start,endpoint_signed_pp=end,true_drop_pp=td,
            predicted_drop_pp=pd,change_signed_pp=change,drop_abs_pp=abs(change),
            underdecline=td>=5 and pd<td,severe_underdecline=td>=5 and pd<.3*td,
            late_true_decline_pp_per_100_cycles=true_rate,
            late_pred_decline_pp_per_100_cycles=pred_rate,
            late_underdecline=true_rate>0 and pred_rate<.5*true_rate,
            mape_percent=np.average(np.abs((p-y)/y))*100)
        for k,v in values.items():np.testing.assert_allclose(r[k],v,rtol=0,atol=1e-10)
        np.testing.assert_allclose(end,start+change,rtol=0,atol=1e-10)
        with np.load(child/f"candidate_{r['candidate']}"/'training_predictions.npz') as a:
            maximum=max(float(a[k+'__100__cycle'][-1]) for k in plan['train'])
        assert r['training_max_cycle']==maximum and r['query_end_beyond_training_max']==bool(c[-1]>maximum)
    groups=[(summary['arms'][a],[r for r in rows if r['arm']==a]) for a in ('selected','control')]
    groups += [(summary['by_fold'][p['name']],[r for r in rows if r['arm']=='selected' and r['fold']==p['name']]) for p in pc['plans']]
    for saved,subset in groups:
        assert saved['count']==len(subset)
        for k,v in saved.items():
            if k=='count':continue
            if k=='mean_start_abs_pp':values=[abs(r['start_signed_pp']) for r in subset]
            elif k=='mean_endpoint_abs_pp':values=[abs(r['endpoint_signed_pp']) for r in subset]
            elif k=='mean_change_signed_pp':values=[r['change_signed_pp'] for r in subset]
            elif k=='drop_dominates_start_fraction':values=[r['drop_abs_pp']>abs(r['start_signed_pp']) for r in subset]
            else:values=[r[k] for r in subset]
            np.testing.assert_allclose(v,np.average(values),rtol=0,atol=1e-10)
    assert summary['unique_validation_cells']==len({r['cell_id'] for r in rows})==81
    save_json(root/'independent_verification.json',dict(passed=True,rows=len(rows),
        unique_validation_cells=81,outer_predictions_read=False,new_fits=0,
        scope='all diagnostic rows, frozen inputs, selection, arithmetic and summaries; not models or causal claims',
        verifier_sha256=sha(__file__),goal_complete=False))
    print(json.dumps(dict(passed=True,rows=len(rows),unique_validation_cells=81)),flush=True)


if __name__=='__main__':main()
