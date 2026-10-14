"""Stream raw source cycles; early V/I only, capacity strictly label-only."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import shutil
import traceback
import zipfile
import joblib
import numpy as np
import pandas as pd
from scripts.prepare_same_type_forecast import save_json,sha

FAMILY='Zhu2022_dataset1_NCA_18650_3p5Ah_cohort'
COLS=['time/s','Ecell/V','<I>/mA','cycle number','Q discharge/mA.h']

def signal_features(group):
    # Explicit signal allowlist; no Q, capacity, SOH or full-cycle duration.
    signal=group[['time/s','Ecell/V','<I>/mA']].to_numpy(dtype=float)
    take=signal[:,2]<-100.
    if not take.any():raise ValueError('no measured discharge current')
    start=int(np.flatnonzero(take)[0]);signal=signal[start:]
    if signal[0,1]<3.8:raise ValueError('non-comparable low-SOC diagnostic discharge start')
    t=signal[:,0]-signal[0,0]
    outside=np.flatnonzero(t>900)
    if len(outside):stop=int(outside[0])
    elif len(t) and t[-1]==900:stop=len(t)
    else:raise ValueError('discharge shorter than fixed900s input window')
    signal=signal[:stop];t=t[:stop]
    if not np.isfinite(signal).all():raise ValueError('invalid early discharge V/I/time')
    if np.any(np.diff(t)<=0):raise ValueError('non-increasing early discharge time')
    if len(t)<3:raise ValueError('insufficient early discharge samples')
    grid=np.arange(0.,901.,90.)
    def stats(a):return np.r_[np.interp(grid,t,a),a.mean(),a.std(),a.min(),a.max(),a[-1]-a[0]]
    return np.r_[stats(signal[:,1]),stats(signal[:,2]/1000.)]

def capacity_label(group):
    a=group[COLS].to_numpy(dtype=float)
    if not np.isfinite(a).all():raise ValueError('nonfinite capacity-audit waveform')
    t,v,current,q=a[:,0],a[:,1],a[:,2],a[:,4]
    negative=np.flatnonzero(current<-100.)
    if len(negative)<3:raise ValueError('no complete measured discharge')
    start,end=int(negative[0]),int(negative[-1])
    if v[:start+1].max()<4.18 or v[start]<3.8 or v[negative].min()>2.70:
        raise ValueError('non-comparable full-charge/cutoff discharge protocol; partial diagnostic')
    peak=float(q.max())
    if q.min()<-1e-5 or not 0<peak<=4200:raise ValueError('invalid finite/physical discharge Q')
    if np.any(np.diff(t[start:end+1])<0):raise ValueError('time reversal in discharge label')
    integrated=float(np.trapezoid(np.maximum(-current[start:end+1],0),t[start:end+1])/3600)
    if abs(integrated-peak)>max(10.,.05*peak):raise ValueError('capacity inconsistent with measured current integral')
    return peak

def completed_groups(reader):
    carry=None
    for chunk in reader:
        frame=pd.concat([carry,chunk],ignore_index=True) if carry is not None else chunk
        if frame['cycle number'].isna().any():raise ValueError('nonfinite source cycle')
        if np.any(np.diff(frame['cycle number'].to_numpy())<0):raise ValueError('nonmonotone source cycle sequence')
        last=frame['cycle number'].iloc[-1]
        carry=frame[frame['cycle number']==last].copy()
        for n,group in frame[frame['cycle number']!=last].groupby('cycle number',sort=False):yield n,group
    if carry is not None:
        for n,group in carry.groupby('cycle number',sort=False):yield n,group

def prefix(cycle,raw,conditions):
    cycle=np.asarray(cycle,dtype=float);raw=np.asarray(raw,dtype=float)
    if len(cycle)<3:raise ValueError('insufficient early source-cycle signals')
    first=np.median(raw[:min(5,len(raw))],axis=0);last=np.median(raw[-3:],axis=0)
    take=cycle>=cycle[-1]-9;t=cycle[take]-cycle[take].mean();x=raw[take]
    slope=np.sum(t[:,None]*(x-x.mean(axis=0)),axis=0)/np.sum(t*t)*100 if np.sum(t*t)>0 else np.zeros(raw.shape[1])
    return np.r_[first,last,last-first,slope,conditions,cycle[0]/100,cycle[-1]/100,len(cycle)/100]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--raw-dir',type=Path,required=True);args=parser.parse_args()
    manifest=json.loads((args.raw_dir/'intake_manifest.json').read_text());assert manifest['complete']
    out=Path('data/interim/zhu2022_nca')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True)
    shutil.copy2(__file__,out/Path(__file__).name)
    save_json(out/'config.json',dict(source_sha256=sha(__file__),raw_sha256=manifest['sha256'],origin=100,
        fixed_input_seconds=900,capacity_input=False,nominal_capacity_mAh=3500,
        label_rules='full-charge4.18V, loaded discharge start>=3.8V, discharge cutoff<=2.70V; finite0<Q<=4200mAh; Q/current integral consistency5% or10mAh; never2500mAh lower cutoff',
        task_rule='need valid early signals plus>=2 future labels; report gaps and ineligible cells, never silently discard',
        family=FAMILY,model_manufacturer='unverified; separate original experimental cohort',license=manifest['license']))
    try:
        archive=args.raw_dir/manifest['file'];assert sha(archive)==manifest['sha256']
        cells={};audit=[];excluded=[];ineligible=[]
        with zipfile.ZipFile(archive) as z:
            files=sorted(i.filename for i in z.infolist() if i.filename.lower().endswith('.csv') and not i.is_dir())
            for name in files:
                key=Path(name).stem
                match=re.fullmatch(r'CY(\d+)-(025|05|1)_1-#(\d+)',key)
                if not match:raise ValueError('unresolved experimental condition filename: '+key)
                temp=float(match[1]);charge={'025':.25,'05':.5,'1':1.}[match[2]]
                early_cycles=[];early_raw=[];label_cycles=[];capacity=[];source_cycles=[];previous=None
                structural_error=None
                with z.open(name) as handle:
                    reader=pd.read_csv(handle,usecols=COLS,chunksize=200000)
                    groups=completed_groups(reader)
                    while True:
                        try:n,g=next(groups)
                        except StopIteration:break
                        except ValueError as e:
                            structural_error=str(e);break
                        if not np.isfinite(n) or n!=int(n):raise ValueError('invalid integer source cycle')
                        n=int(n)
                        if previous is not None and n<=previous:raise ValueError('duplicate or nonmonotone cycle group')
                        previous=n;source_cycles.append(n)
                        try:
                            value=capacity_label(g);label_cycles.append(n);capacity.append(value)
                        except ValueError as e:excluded.append(dict(cell_id=key,cycle=n,kind='label',reason=str(e)))
                        if n<=100:
                            try:
                                features=signal_features(g)
                                poisoned=g.copy();poisoned['Q discharge/mA.h']=np.inf
                                np.testing.assert_array_equal(features,signal_features(poisoned))
                                early_cycles.append(n);early_raw.append(features)
                            except ValueError as e:excluded.append(dict(cell_id=key,cycle=n,kind='input',reason=str(e)))
                if structural_error is not None:
                    reason='ambiguous restarted/nonmonotone source sequence: '+structural_error
                    audit.append(dict(cell_id=key,member=name,eligible=False,reason=reason,
                        temperature_C=temp,charge_rate_C=charge,discharge_rate_C=1.))
                    ineligible.append(dict(cell_id=key,reason=reason))
                    print(key,'eligible False:',reason,flush=True)
                    continue
                label_cycles=np.asarray(label_cycles);capacity=np.asarray(capacity)
                reference_cycle=int(label_cycles[0]) if len(label_cycles) else None
                future=label_cycles>100
                gaps=[int(n) for n in range(1,min(100,max(source_cycles))+1) if n not in early_cycles]
                report=dict(cell_id=key,member=name,source_cycles=len(source_cycles),source_cycle_min=min(source_cycles),
                    source_cycle_max=max(source_cycles),temperature_C=temp,charge_rate_C=charge,discharge_rate_C=1.,
                    early_input_cycles=len(early_cycles),early_input_gaps=gaps,valid_labels=len(label_cycles),
                    future_labels=int(future.sum()),reference_source_cycle=reference_cycle,
                    initial_capacity_mAh=float(capacity[0]) if len(capacity) else None)
                try:
                    if reference_cycle is None or reference_cycle>100:raise ValueError('reference label not established by origin100')
                    if future.sum()<2:raise ValueError('insufficient observed future labels after source cycle100')
                    if len(early_cycles)<90:raise ValueError('fewer than90 valid early source-cycle signals under fixed coverage rule')
                    features=prefix(early_cycles,early_raw,[temp,charge,1.,2.65,4.2])
                    cells[key]=dict(prefix=features,cycle=label_cycles[future],truth=capacity[future]/capacity[0],
                        physical_family=FAMILY,early_cycle=np.asarray(early_cycles),early_raw=np.asarray(early_raw),
                        conditions=np.array([temp,charge,1.,2.65,4.2]))
                    report['eligible']=True
                except ValueError as e:report.update(eligible=False,reason=str(e));ineligible.append(dict(cell_id=key,reason=str(e)))
                audit.append(report);print(key,'source_cycles',len(source_cycles),'eligible',report['eligible'],flush=True)
        joblib.dump(cells,out/'prepared_cells.joblib')
        save_json(out/'audit.json',dict(complete=True,raw_cells=len(files),eligible_cells=len(cells),cells=audit,
            record_exclusions=excluded,ineligible_cells=ineligible,raw_sha256=manifest['sha256'],
            capacity_input_poison_check=True,early_input_coverage_rule='>=90 valid cycles <=100',
            source_conditions='filename and paper Table1',physical_family=FAMILY,
            measured_temperature_available=False,current_units='raw mA converted to A',
            manufacturer_model='unverified; no cross-cohort physical-family pooling',
            reference_policy='first valid comparable same-cell discharge capacity, label only',
            raw_directory=str(args.raw_dir.resolve()),task='futureSOH from first100 source-cycle V/I'))
        print('OUTPUT='+str(out.resolve()),'eligible='+str(len(cells)),flush=True)
    except BaseException:
        save_json(out/'failure.json',dict(traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
