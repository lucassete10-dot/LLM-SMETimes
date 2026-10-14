"""TRAIN-only trace/capacity consistency examples; no automatic label changes."""
import json
from pathlib import Path
import h5py
import numpy as np
from prepare_mit import read_ref
from scripts.audit_llm_curve_repair import load_frozen,fingerprint
from scripts.prepare_same_type_forecast import save_json


def main():
    root=Path(json.loads(Path('experiments/llm_curve_repair/latest_P0.json').read_text())['folder'])
    cfg=json.loads((root/'config.json').read_text());manifest,cells,plan,_,_=load_frozen(cfg)
    if 'b1c0' not in cfg['tiny_train_ids'] or 'b1c0' not in plan['train']:raise ValueError('TRAIN-only audit')
    cell=cells['b1c0'];record=next(r for r in manifest['cells'] if r['cell_id']=='b1c0')
    x,y=cell['label_cycle'],cell['soh'];future=np.flatnonzero((x>cfg['origin'])&np.isfinite(y)&(y>0))
    # Selection is solely a TRAIN diagnostic, never a filtering rule.
    worst=future[np.argsort(y[future]-np.interp(x[future],x[future][::25],y[future][::25]))[:5]]
    examples=[]
    with h5py.File(record['raw_source'],'r') as mat:
        group=mat['batch'];summary=mat[group['summary'][0,0]];traces=mat[group['cycles'][0,0]]
        cycles=np.asarray(summary['cycle']).ravel().astype(int);q=np.asarray(summary['QDischarge']).ravel()
        for cycle in sorted(set(int(x[j])+delta for j in worst for delta in (-1,0,1))):
            found=np.flatnonzero(cycles==cycle)
            if not len(found):continue
            index=int(found[0]);t,v,i,qd=[read_ref(mat,traces[k][index,0]) for k in ('t','V','I','Qd')]
            dt=np.diff(t);discharge=i<-.1;indices=np.flatnonzero(discharge)
            if not len(indices):raise ValueError('missing diagnostic discharge')
            adjacent=discharge[:-1]&discharge[1:]&(dt>=0)
            # Source I is C-rate (recorded amperes / fixed nominal 1.1 A).
            ampere=1.1*i
            integral=np.sum(-.5*(ampere[:-1][adjacent]+ampere[1:][adjacent])*dt[adjacent]/60)
            all_integral=np.sum(np.maximum(-.5*(ampere[:-1]+ampere[1:]),0)*np.maximum(dt,0)/60)
            examples.append({'cycle':cycle,'soh':float(q[index]/record['reference_capacity_ah']),
                'summary_qd_ah':float(q[index]),'trace_qd_max_ah':float(qd.max()),
                'discharge_median_current_c_rate':float(np.median(i[discharge])),
                'discharge_median_current_a':float(np.median(ampere[discharge])),
                'discharge_first_voltage_v':float(v[indices[0]]),'discharge_last_voltage_v':float(v[indices[-1]]),
                'discharge_duration_minutes':float(t[indices[-1]]-t[indices[0]]),
                'adjacent_discharge_integral_ah':float(integral),'all_negative_current_integral_ah':float(all_integral),
                'negative_time_step_count':int((dt<0).sum()),'qd_minus_all_integral_ah':float(qd.max()-all_integral)})
    report={'scope':'TRAIN label-provenance audit only; no inference use or automatic label exclusion',
        'cell':'b1c0','source_sha256':fingerprint(Path(__file__)),'raw_source':record['raw_source'],
        'units_source':'https://github.com/chueh-ermon/BMS-autoanalysis/blob/master/cell_analysis.m',
        'nominal_current_scale_to_ampere':1.1,'time_units':'minutes',
        'raw_sha256':fingerprint(record['raw_source']),'examples':examples,
        'interpretation':'Saved summary Qd and trace Qd agree. Some neighboring cycles have near-identical I/time integrals while Qd varies. Integration assumptions, trace processing and author instrument records must be investigated before interpreting the discrepancy as an erroneous label. No reference or label replacement is authorized by this diagnostic.'}
    save_json(root/'TRAIN_b1c0_spike_trace_audit.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
