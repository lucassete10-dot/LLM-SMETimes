"""Prepare verified public HUST signals and nominal-capacity labels, preserving audit."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from scripts.inspect_hust_public import PrimitiveUnpickler
from scripts.prepare_same_type_forecast import save_json,sha


def digest(path):
    h=hashlib.md5()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def features(row):
    t=np.asarray(row['time_in_s'],float);i=np.asarray(row['current_in_A'],float);v=np.asarray(row['voltage_in_V'],float)
    if not (t.shape==i.shape==v.shape and t.ndim==1 and len(t)>2):raise ValueError('signal shape mismatch')
    mask=(i<-.1)&np.isfinite(t)&np.isfinite(i)&np.isfinite(v)
    td=t[mask];ii=i[mask];vv=v[mask]
    if len(td)<3:raise ValueError('discharge signals unavailable')
    td=td-td[0]
    if np.any(np.diff(td)<=0):raise ValueError('discharge time not increasing')
    if td[-1]<300:raise ValueError('less than fixed 300s discharge input')
    out=[]
    for a in (vv,ii):
        grid=np.interp(np.linspace(0,300,11),td,a);take=td<=300
        out.extend(np.r_[grid,np.mean(a[take]),np.std(a[take]),np.min(a[take]),np.max(a[take]),grid[-1]-grid[0]])
    return out


def main():
    source=Path('data/raw/five_dataset_public/BatteryLife_v12/HUST.zip')
    assert digest(source)=='de6b6d1b0b20616fbb96c72f3231c082'
    root=Path('data/interim/hust_public')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True)
    print('RUN='+str(root.resolve()),flush=True);records=[]
    with zipfile.ZipFile(source) as z:
        members=[n for n in z.namelist() if n.endswith('.pkl')];assert len(members)==77
        for name in members:
            with z.open(name) as f:cell=PrimitiveUnpickler(f).load()
            key=cell['cell_id'];nominal=float(cell['nominal_capacity_in_Ah']);assert nominal==1.1
            cycles=cell['cycle_data'];raw=[];early=[];c=[];y=[];excluded=[];observed=[]
            for row in cycles:
                n=float(row['cycle_number']);q=np.asarray(row['discharge_capacity_in_Ah'],float)
                if q.ndim==1 and len(q) and np.isfinite(q).all() and q.max()>0:
                    c.append(n);y.append(float(q.max()/nominal))
                else:excluded.append(dict(cycle=n,role='label',reason='nonfinite or nonpositive discharge capacity'))
                if n<=200:
                    try:raw.append(features(row));early.append(n)
                    except ValueError as e:excluded.append(dict(cycle=n,role='signal',reason=str(e)))
                if n<=10:
                    a=np.asarray(row['current_in_A'],float);observed.append(float(abs(np.min(a))/nominal))
            if np.any(np.diff(c)<=0):raise ValueError('label cycle order '+key)
            condition=[30.,nominal]
            for protocol,size in [('charge_protocol',3),('discharge_protocol',4)]:
                condition.extend([float(p['rate_in_C']) if p.get('rate_in_C') is not None else 0. for p in cell[protocol]])
                if len(cell[protocol])!=size:raise ValueError('unexpected protocol size')
            output=root/(key+'.npz')
            np.savez_compressed(output,early_cycle=np.array(early),early_raw=np.asarray(raw),cycle=np.array(c),truth=np.array(y),conditions=np.array(condition))
            metadata={k:v for k,v in cell.items() if k!='cycle_data'}
            records.append(dict(cell_id=key,path=str(output.resolve()),sha256=sha(output),label_count=len(c),
                cycle_min=min(c),cycle_max=max(c),visible_signal_count=len(early),metadata=metadata,
                exclusions=excluded,observed_first10_peak_discharge_C=observed,physical_family='HUST_LFP_18650_1p1Ah_model_unverified'))
            print(f'CELL {key} labels={len(c)} early={len(early)} excluded={len(excluded)}',flush=True)
    save_json(root/'index.json',dict(cells=records,source=str(source.resolve()),publisher_md5_verified=True,
        source_md5='de6b6d1b0b20616fbb96c72f3231c082',label_definition='cycle discharge capacity / rated 1.1Ah',
        signal_contract='first 300 seconds of discharge in each visible cycle; V/I only; no capacity inputs',
        input_cache_max_cycle=200,temperature_measured=False,nominal_temperature_C=30,
        protocol_note='77 discharge protocols; held-out-cell prediction is not automatically same-protocol prediction',
        limitations=['Processed mirror, raw author format not audited','Operational capacity depends on discharge protocol',
                     'Metadata C-rates must be reconciled with recorded currents before same-type claims'],goal_complete=False))
    print('COMPLETE '+str(root.resolve()),flush=True)


if __name__=='__main__':main()
