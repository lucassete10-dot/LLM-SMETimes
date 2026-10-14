"""Rebuild author-supported cumulative cycling coordinates without future fill."""
from datetime import datetime,timezone
import hashlib,json,pickle
from pathlib import Path
import numpy as np
from scripts.prepare_same_type_forecast import sha,save_json


def main():
    source=Path('data/raw/five_dataset_public/UConn_author')
    pulse=source/'processed_data/UConn-ILCC-LFP/data_slowpulse_1.pkl'
    manifest=json.loads((source/'manifest.json').read_text())
    expected=next(r['sha256'] for r in manifest['items'] if r['path'].endswith('data_slowpulse_1.pkl'))
    assert sha(pulse)==expected
    overview=source/'notebooks/data_overview.ipynb'
    notebook=json.loads(overview.read_text(encoding='utf-8'))
    code=''.join(notebook['cells'][3]['source'])
    assert "np.cumsum(cycles)" in code
    provenance=json.loads((source/'overview_provenance.json').read_text())
    assert sha(overview)==provenance['sha256']
    # Only the pinned, previously authenticated author pickle is deserialized.
    data=pickle.loads(pulse.read_bytes())
    root=Path('data/interim/uconn_causal_cycles')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True); records=[]; gaps=[]; origins=[10,20,30,50,75,100,150,200]
    for cid in np.unique(data['cell_id']):
        idx=np.flatnonzero(data['cell_id']==cid); rpts=np.unique(data['rpt'][idx])
        cumulative=0.; last_rpt=None; c=[]; v=[]; q=[]; rpt_ids=[]
        group=np.unique(data['group_id'][idx]); assert len(group)==1
        for rpt in rpts:
            selected=idx[data['rpt'][idx]==rpt]
            intervals=np.unique(data['num_cycles'][selected])
            if len(intervals)!=1 or not np.isfinite(intervals[0]):
                gaps.append(dict(cell_id=int(cid),rpt=int(rpt),reason='unknown interval cycles',
                    action='stop coordinate-dependent data at first unknown interval; no next-week filling',not_bad_cell=True))
                break
            if (last_rpt is None and rpt!=0) or (last_rpt is not None and rpt!=last_rpt+1):
                gaps.append(dict(cell_id=int(cid),rpt=int(rpt),reason='missing RPT interval',not_bad_cell=True)); break
            increment=float(intervals[0]); assert increment>=0 and (rpt!=0 or increment==0)
            cumulative+=increment; last_rpt=int(rpt)
            profile=[]
            for soc in (20,50,90):
                for kind in ('chg','dchg'):
                    row=selected[(data['soc'][selected]==soc)&(data['pulse_type'][selected]==kind)]
                    assert len(row)==1
                    profile.extend(np.asarray(data['voltage'][row[0]],float))
            capacities=np.unique(data['q_dchg'][selected])
            assert len(capacities)==1 and np.isfinite(capacities[0]) and capacities[0]>0
            c.append(cumulative); v.append(profile); q.append(float(capacities[0])); rpt_ids.append(int(rpt))
        c=np.array(c,float); assert len(c)>=3 and np.all(np.diff(c)>0)
        path=root/f'cell_{int(cid):03d}.npz'
        np.savez_compressed(path,source_cycle=c,rpt=np.array(rpt_ids),pulse_voltage=np.array(v),capacity_ah=np.array(q),
            conditions=np.array([int(group[0])]),soh_initial=np.array(q)/q[0])
        records.append(dict(cell_id=f'UConn_LFP_{int(cid):03d}',author_cell_id=int(cid),group_id=int(group[0]),
            physical_family='LithiumWerks_APR18650M1B_LFP_graphite',path=str(path.resolve()),sha256=sha(path),
            valid_rpts=len(c),last_known_source_cycle=float(c[-1]),initial_capacity_ah=float(q[0]),
            prefix_observations={str(n):int(np.sum(c<=n)) for n in origins},
            input_schema='Six known-protocol pulse voltage profiles (3 SOC x 2 signs), 101 time points each; NO measured current/temperature in this processed file',
            forbidden_inputs_not_used=['q_dchg','soh','soc - coulomb','dcir capacity-derived dependencies'],
            training_ready=False))
    assert len(records)==64
    audit=dict(cells=records,coordinate_gaps=gaps,source_sha256=sha(pulse),author_notebook_sha256=sha(overview),
        coordinate_definition='Cumulative sum of per-RPT aging-cycle intervals, as explicitly used by author notebook cell 3.',
        interpolation_of_intervals=False,future_label_fill=False,source_cycles_are_not_RPT_counts=True,
        ready=False,remaining='Verify nominal capacity/protocol mapping and implement sparse-prefix model; author initial-normalized labels retained only for audit.',
        goal_complete=False)
    save_json(root/'index.json',audit)
    print(root.resolve()); print('Cells:',len(records),'coordinate gap truncations:',len(gaps))
    print('Counts of RPT observations per input length:',{str(n):dict(zip(*np.unique([r['prefix_observations'][str(n)] for r in records],return_counts=True))) for n in origins})


if __name__=='__main__':main()
