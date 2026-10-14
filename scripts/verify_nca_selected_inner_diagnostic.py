"""Recompute diagnostic arithmetic without importing its decomposition helper."""
import json,sys
from pathlib import Path
import numpy as np
from scripts.prepare_same_type_forecast import sha,save_json


def main(root):
    cfg=json.loads((root/'config.json').read_text());rows=json.loads((root/'rows.json').read_text())
    choices=json.loads((root/'choices.json').read_text());expected=[]
    assert sha(Path(cfg['parent'])/'config.json')==cfg['parent_config_sha256']
    for f,digest in cfg['input_sha256'].items():assert sha(Path(f))==digest
    for f,digest in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/Path(f).name)==digest
    assert len(choices)==25 and len(rows)==125
    for choice in choices:
        plan=next(p for p in cfg['plans'] if p['name']==choice['fold'])
        assert not set(plan['train'])&set(plan['validation']) and not (set(plan['train'])|set(plan['validation']))&set(plan['test'])
        file=Path(choice['selection_file']).parent/f"candidate_{choice['candidate']}"/'validation_predictions.npz'
        with np.load(file,allow_pickle=False) as a:
            for k in plan['validation']:
                c,y,p=[a[k+'__100__'+f].astype(float) for f in ('cycle','truth','prediction')]
                start=100*(p[0]-y[0]);end=100*(p[-1]-y[-1])
                td=100*(y[0]-y[-1]);pd=100*(p[0]-p[-1]);delta=td-pd
                mask=c>=c[0]+.75*(c[-1]-c[0])
                if mask.sum()<2:mask[-2:]=True
                rates=[-float(np.polyfit(c[mask]-c[mask].mean(),v[mask],1)[0])*10000 for v in (y,p)]
                vals=dict(start_signed_pp=start,endpoint_signed_pp=end,true_drop_pp=td,predicted_drop_pp=pd,
                    change_signed_pp=delta,drop_abs_pp=abs(delta),underdecline=bool(td>=5 and pd<td),
                    severe_underdecline=bool(td>=5 and pd<.3*td),late_true_decline_pp_per_100_cycles=rates[0],
                    late_pred_decline_pp_per_100_cycles=rates[1],late_underdecline=bool(rates[0]>0 and rates[1]<.5*rates[0]),
                    mape_percent=float(np.average(np.abs(p-y)/y)*100))
                row=rows[len(expected)];assert (row['cell_id'],row['fold'],row['seed'],row['mode'])==(k,choice['fold'],choice['seed'],choice['mode'])
                for f,v in vals.items():np.testing.assert_allclose(row[f],v,rtol=0,atol=1e-9)
                np.testing.assert_allclose(end,start+delta,atol=1e-10,rtol=0)
                expected.append(vals)
    s=json.loads((root/'summary.json').read_text())
    measures=dict(mean_start_abs_pp=np.mean([abs(r['start_signed_pp']) for r in expected]),
        mean_endpoint_abs_pp=np.mean([abs(r['endpoint_signed_pp']) for r in expected]),
        mean_drop_error_pp=np.mean([r['drop_abs_pp'] for r in expected]),
        mean_signed_change_pp=np.mean([r['change_signed_pp'] for r in expected]),
        underdecline_fraction=np.mean([r['underdecline'] for r in expected]),
        severe_underdecline_fraction=np.mean([r['severe_underdecline'] for r in expected]),
        late_underdecline_fraction=np.mean([r['late_underdecline'] for r in expected]),
        drop_dominates_start_fraction=np.mean([r['drop_abs_pp']>abs(r['start_signed_pp']) for r in expected]))
    for f,v in measures.items():np.testing.assert_allclose(s[f],v,rtol=0,atol=1e-9)
    assert s['records']==125 and s['unique_validation_cells']==len({r['cell_id'] for r in rows})==22
    save_json(root/'independent_verification.json',dict(passed=True,records=125,choices=25,
        arithmetic='Separate endpoint/drop formula and polyfit late slopes; source hashes checked.',
        selection_replay='Producer checks saved-model replay; this verifier checks arithmetic and inputs, not retraining.',
        outer_loaded=False,new_fits=0,verifier_sha256=sha(Path(__file__))))
    print('PASS: 125 inner diagnostic records, 22 unique validation cells; no outer reads.')


if __name__=='__main__':main(Path(sys.argv[1]))
