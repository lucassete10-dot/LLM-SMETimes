"""Restore source-cycle<=200 V/I/T diagnostics for development cells only."""
from datetime import datetime,timezone
import json,traceback
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from prepare_xjtu import diagnostic,stage_features,FEATURE_NAMES
from scripts.train_curve_fidelity import AUDIT
from scripts.prepare_same_type_forecast import sha,save_json


def main():
    plan=next(p for p in json.loads((AUDIT/'splits.json').read_text())['plans'] if p['name']=='XJTU_fold1')
    dev=sorted(plan['train']+plan['validation']);registry=json.loads((AUDIT/'cell_registry.json').read_text())['cells']
    assert not set(dev)&set(plan['test'])
    root=Path('data/interim/xjtu_extended_signals')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True)
    records=[];print('RUN='+str(root.resolve()),flush=True)
    try:
        for cid in dev:
            source=Path(registry[cid]['raw_source']);batch=int(cid.split('_')[0][1:]);data=loadmat(source,simplify_cells=True)['data']
            cycles=[];features=[];excluded=[]
            for j,row in enumerate(data[:200]):
                if not diagnostic(batch,row['description']):continue
                # Capacity, SOH and end-of-life fields are never supplied to the feature extractor.
                signal={k:row[k] for k in ['relative_time_min','voltage_V','current_A','temperature_C']}
                charge=stage_features(signal,0);discharge=stage_features(signal,2)
                if charge is None or discharge is None:
                    excluded.append(dict(source_cycle=j+1,reason='insufficient finite 120-second diagnostic signals',not_bad_cell=True));continue
                cycles.append(j+1);features.append(np.r_[charge,discharge])
            assert len(cycles)>=3 and np.all(np.diff(cycles)>0)
            path=root/(cid+'.npz');np.savez_compressed(path,signal_cycle=np.array(cycles),signal_raw=np.array(features))
            # Compare overlapping visible rows against the original prepared signal values.
            with np.load(registry[cid]['prepared_file'],allow_pickle=False) as old:
                oc,ox=old['cycle'],old['raw_features'];count=0
                for c,x in zip(cycles,features):
                    ii=np.flatnonzero(oc==c)
                    if len(ii):np.testing.assert_allclose(x,ox[ii[0]],rtol=1e-5,atol=1e-6,equal_nan=True);count+=1
            records.append(dict(cell_id=cid,path=str(path.resolve()),sha256=sha(path),source=str(source),source_sha256=sha(source),
                observations=len(cycles),overlap_rows_verified=count,exclusions=excluded))
            save_json(root/'progress.json',dict(completed_cells=len(records),planned_cells=len(dev),terminal=False))
            print('CELL',cid,len(cycles),'overlap',count,flush=True)
        save_json(root/'index.json',dict(dataset='XJTU',cells=records,feature_names=FEATURE_NAMES,max_source_cycle=200,
            capacity_used_for_signal_eligibility=False,outer_loaded=False,development=dev,reserve=plan['test'],
            source_sha256={str(p):sha(p) for p in [Path(__file__),Path('prepare_xjtu.py')]},goal_complete=False))
        save_json(root/'progress.json',dict(completed_cells=len(records),planned_cells=len(dev),terminal=True,success=True))
    except BaseException:
        save_json(root/'failure.json',dict(traceback=traceback.format_exc(),completed_records=records))
        save_json(root/'progress.json',dict(completed_cells=len(records),terminal=True,success=False));raise


if __name__=='__main__':main()
