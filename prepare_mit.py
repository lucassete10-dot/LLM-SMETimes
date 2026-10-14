"""Read Severson Batch 1 HDF5 MAT, preserving cell and cycle boundaries.

Input features use only first 300 seconds of the main 4C discharge. Full-cycle
Qd is a target/quality check only, never a predictor.
"""
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

SOURCE=Path(r'D:\LLM\MIT\raw\2017-05-12_batchdata_updated_struct_errorcorrect.mat')
OUT=Path(r'D:\LLM\MIT\processed')
GRID=np.arange(0,301,30)
COMMON_GRID=np.array([0.,100.,200.,300.])
COMMON_NAMES=([f'voltage_v_t{int(t):04d}s' for t in COMMON_GRID]+[
    f'current_a_t{int(t):04d}s' for t in COMMON_GRID]+[
    'voltage_drop_v','voltage_mean_v','voltage_std_v','current_abs_mean_a',
    'current_std_a','prefix_charge_ah','prefix_energy_wh'])
ENHANCED_NAMES=([f'voltage_v_t{int(t):04d}s' for t in GRID]+[
    f'current_a_t{int(t):04d}s' for t in GRID]+[
    f'temperature_c_t{int(t):04d}s' for t in GRID]+[
    'voltage_drop_v','voltage_mean_v','voltage_std_v','current_abs_mean_a',
    'current_std_a','prefix_charge_ah','prefix_energy_wh',
    'voltage_slope_first_v_per_s','voltage_slope_last_v_per_s',
    'voltage_curvature_v','voltage_median_v','voltage_iqr_v',
    'voltage_integral_vs','current_median_a','current_iqr_a',
    'temperature_rise_c','temperature_median_c','temperature_iqr_c'])


def read_ref(f,ref):
    return np.asarray(f[ref]).reshape(-1)


def intervals(mask):
    positions=np.flatnonzero(mask)
    if not len(positions):return []
    return np.split(positions,np.flatnonzero(np.diff(positions)>1)+1)


def discharge_slice(t,i,v):
    """Find the principal 4C constant-current discharge segment."""
    candidates=[]
    for run in intervals(i<-2):
        if len(run)<3:continue
        duration=(t[run[-1]]-t[run[0]])*60
        if duration>=300 and np.isfinite(v[run]).all():
            candidates.append((duration,run))
    if not candidates:return None
    return max(candidates,key=lambda p:p[0])[1]


def feature_vector(t_minutes,current,voltage,temperature,grid):
    """No interpolation may read a measurement after t=300 s."""
    t=(t_minutes-t_minutes[0])*60
    # Instrument step boundaries can reverse by ~1e-11 s from roundoff.
    # Preserve rejection of actual time reversals, while keeping valid records.
    if np.any(np.diff(t) < -1e-6):
        raise ValueError('short or nonmonotone prefix')
    t=np.maximum.accumulate(t)
    take=t<=300
    t,v,i,T=t[take],voltage[take],current[take],temperature[take]
    if len(t)<3 or t[0]>30 or t[-1]<285 or np.any(np.diff(t)<0):
        raise ValueError('short or nonmonotone prefix')
    vg=np.interp(grid,t,v)
    ig=np.interp(grid,t,i)
    Tg=np.interp(grid,t,T) if np.isfinite(T).all() else np.full(len(grid),np.nan)
    pt=np.r_[t,300]
    pv=np.r_[v,vg[-1]]
    pi=np.r_[i,ig[-1]]
    basic=[vg[0]-vg[-1],pv.mean(),pv.std(),np.abs(pi).mean(),pi.std(),
           np.trapezoid(np.abs(pi),pt)/3600,np.trapezoid(np.abs(pi)*pv,pt)/3600]
    if len(grid)==4:
        return np.r_[vg,ig,basic].astype('float32')
    more=[(vg[3]-vg[0])/90,(vg[-1]-vg[-4])/90,
          vg[0]-2*vg[len(vg)//2]+vg[-1],
          np.median(vg),np.subtract(*np.percentile(vg,[75,25])),
          np.trapezoid(vg,grid),np.median(ig),np.subtract(*np.percentile(ig,[75,25])),
          Tg[-1]-Tg[0],np.nanmedian(Tg) if np.isfinite(Tg).any() else np.nan,
          np.subtract(*np.nanpercentile(Tg,[75,25])) if np.isfinite(Tg).any() else np.nan]
    return np.r_[vg,ig,Tg,basic,more].astype('float32')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    audit=[]
    with h5py.File(SOURCE,'r') as f:
        batch=f['batch']
        n=len(batch['summary'])
        for cell_idx in range(n):
            name=f'b1c{cell_idx}'
            summary=f[batch['summary'][cell_idx,0]]
            cycles=f[batch['cycles'][cell_idx,0]]
            capacity=np.asarray(summary['QDischarge']).reshape(-1)
            cycle_number=np.asarray(summary['cycle']).reshape(-1)
            if len(capacity)!=len(cycles['I']):raise ValueError(name+' Qd/cycle count mismatch')
            records=[];common=[];enhanced=[]
            for j in range(len(capacity)):
                q=float(capacity[j]);cy=int(cycle_number[j]);reason=''
                if not np.isfinite(q) or q<=.2:reason='invalid_capacity'
                elif q>1.25:reason='capacity_above_physical_screen_1p25Ah'
                if reason:
                    audit.append({'cell':name,'cycle':cy,'capacity_ah':q,'reason':reason})
                    continue
                i=read_ref(f,cycles['I'][j,0]);v=read_ref(f,cycles['V'][j,0]);t=read_ref(f,cycles['t'][j,0]);T=read_ref(f,cycles['T'][j,0]);qd=read_ref(f,cycles['Qd'][j,0])
                if not(len(i)==len(v)==len(t)==len(T)==len(qd)) or not np.isfinite(np.c_[i,v,t,qd]).all():
                    reason='invalid_curve'
                elif abs(np.nanmax(qd)-q)>max(.02,.03*q):
                    reason='summary_curve_capacity_mismatch'
                else:
                    run=discharge_slice(t,i,v)
                    if run is None:reason='no_300s_main_discharge'
                    else:
                        try:
                            base=feature_vector(t[run],i[run],v[run],T[run],COMMON_GRID)
                            rich=feature_vector(t[run],i[run],v[run],T[run],GRID)
                        except ValueError:
                            reason='bad_prefix'
                if not reason:
                    records.append((cy,q,len(run),float((t[run[-1]]-t[run[0]])*60)))
                    common.append(base);enhanced.append(rich)
                audit.append({'cell':name,'cycle':cy,'capacity_ah':q,'reason':reason})
            if len(records)<50:
                print(name,'SKIP insufficient valid cycles',len(records),flush=True)
                continue
            x=np.stack(common);z=np.stack(enhanced);q=np.array([r[1] for r in records],dtype='float32')
            reference=np.where(np.isfinite(z[0]),z[0],0)
            np.savez_compressed(OUT/f'{name}.npz',common=x-x[0],common_raw=x,
                enhanced=z-reference,enhanced_raw=z,capacity=q,soh=q/q[0],
                cycle=np.array([r[0] for r in records]),common_names=COMMON_NAMES,
                enhanced_names=ENHANCED_NAMES)
            print(name,'kept',len(records),'of',len(capacity),'SOH',round(float(q.min()/q[0]),3),flush=True)
    frame=pd.DataFrame(audit)
    frame.to_csv(OUT/'audit.csv',index=False)
    digest=hashlib.sha256()
    with SOURCE.open('rb') as handle:
        for block in iter(lambda:handle.read(2**20),b''):digest.update(block)
    report={'source':str(SOURCE),'sha256':digest.hexdigest(),
            'input_seconds':300,'capacity_target_only':True,'n_cells':n,
            'saved_cells':len(list(OUT.glob('b1c*.npz'))),
            'valid_cycles':int(frame.reason.eq('').sum()),
            'excluded_reasons':frame[frame.reason.ne('')].reason.value_counts().to_dict(),
            'missing_temperature':'NaNs retained for train-only imputation; common model omits temperature'}
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2))
    print('MANIFEST',report,flush=True)


if __name__=='__main__':main()
