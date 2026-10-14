"""Read-only extraction from original CALCE Excel; no label-based smoothing."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(r'D:\LLM\CALCE\raw\extracted')
OUT = Path(r'D:\LLM\CALCE\processed')
IDS = ['CS2_35','CS2_36','CS2_37','CS2_38']
COLS = ['Date_Time','Test_Time(s)','Step_Time(s)','Step_Index','Cycle_Index',
        'Current(A)','Voltage(V)','Discharge_Capacity(Ah)']
NAMES = ([f'voltage_v_t{t:04d}s' for t in range(0,1001,100)] +
         [f'current_a_t{t:04d}s' for t in range(0,1001,100)] +
         ['voltage_drop_v','voltage_mean_v','voltage_std_v','current_abs_mean_a',
          'current_std_a','prefix_charge_ah','prefix_energy_wh'])


def vi_features(time, voltage, current):
    """Ignore all sensor samples strictly after 1000 s, including interpolation."""
    mask = time <= 1000
    t, v, i = time[mask], voltage[mask], current[mask]
    if len(t) < 3 or t[0] > 60 or np.any(np.diff(t) < 0):
        raise ValueError('Invalid prefix')
    grid = np.arange(0,1001,100)
    vg, ig = np.interp(grid,t,v), np.interp(grid,t,i)
    pt, pv, pi = np.r_[t,1000], np.r_[v,vg[-1]], np.r_[i,ig[-1]]
    return np.r_[vg,ig,vg[0]-vg[-1],pv.mean(),pv.std(),np.abs(pi).mean(),pi.std(),
                 np.trapezoid(np.abs(pi),pt)/3600,np.trapezoid(np.abs(pi)*pv,pt)/3600].astype('float32')


def prepare(cell):
    OUT.mkdir(parents=True,exist_ok=True)
    files = sorted((ROOT/cell).rglob('*.xlsx'))
    frames, sources = [], []
    for idx,path in enumerate(files):
        excel = pd.ExcelFile(path)
        sheets = [s for s in excel.sheet_names if s.startswith('Channel')]
        if not sheets:
            raise ValueError(f'No channel sheet: {path}')
        for sheet in sheets:
            df = pd.read_excel(excel,sheet_name=sheet,usecols=lambda c:c in COLS)
            if set(COLS)-set(df):
                raise ValueError(f'Missing columns: {path}')
            # Arbin capacity is cumulative across cycles within each test file.
            # Use the pre-discharge cycle baseline, not the final cumulative value.
            df['capacity_baseline'] = df.groupby('Cycle_Index')['Discharge_Capacity(Ah)'].transform('min')
            # Step 7 is the documented/observed principal constant-current discharge.
            df = df[(df['Step_Index']==7) & (df['Current(A)'] < -.5)].copy()
            df['file'] = path.name
            frames.append(df)
            sources.append({'file': str(path), 'sheet':sheet, 'rows_discharge':len(df),
                            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        excel.close()
        if idx % 5 == 0:
            print(cell, 'workbooks', idx+1, '/',len(files),flush=True)
    df = pd.concat(frames,ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(subset=COLS).sort_values('Date_Time',kind='stable').reset_index(drop=True)
    cuts = ((df['Cycle_Index'].diff()!=0) | (df['Step_Time(s)'].diff()<=0) |
            (df['Date_Time'].diff().dt.total_seconds()>120))
    segments = cuts.cumsum()
    records, features, audit = [], [], []
    for number,(_,g) in enumerate(df.groupby(segments,sort=False),1):
        t=g['Step_Time(s)'].to_numpy(float)
        v=g['Voltage(V)'].to_numpy(float)
        i=g['Current(A)'].to_numpy(float)
        q=g['Discharge_Capacity(Ah)'].to_numpy(float)-float(g['capacity_baseline'].iloc[0])
        reason = ''
        if not np.isfinite(np.c_[t,v,i,q]).all(): reason='nonfinite'
        elif t[0]>60: reason='missing_start'
        elif t[-1]<1000: reason='shorter_than_1000s'
        elif v[-1]>2.85: reason='incomplete_discharge'
        elif q[-1]<=0: reason='invalid_capacity'
        integral=(np.trapezoid(np.abs(i),t)+abs(i[0])*t[0])/3600
        discrepancy=abs(integral-q[-1])/max(q[-1],1e-8)
        if not reason and discrepancy>.03: reason='capacity_integral_mismatch'
        row={'cell':cell,'observation':number,'source_cycle':int(g['Cycle_Index'].iloc[0]),
             'date':str(g['Date_Time'].iloc[0]),'source_file':str(g['file'].iloc[0]),
             'capacity_ah':float(q[-1]),'duration_s':float(t[-1]),'start_s':float(t[0]),
             'end_voltage_v':float(v[-1]),'integral_capacity_ah':float(integral),
             'capacity_discrepancy_fraction':float(discrepancy),'excluded_reason':reason}
        audit.append(row)
        if not reason:
            features.append(vi_features(t,v,i))
            records.append(row)
    pd.DataFrame(audit).to_csv(OUT/f'{cell}_audit.csv',index=False)
    if len(records)<50:
        print(pd.Series([r['excluded_reason'] for r in audit]).value_counts(),flush=True)
        raise ValueError(f'{cell}: only {len(records)} valid cycles')
    x=np.stack(features)
    capacity=np.array([r['capacity_ah'] for r in records],dtype='float32')
    np.savez_compressed(OUT/f'{cell}.npz',raw_features=x,features=x-x[0],
                        capacity=capacity,soh=capacity/capacity[0],
                        observation=np.array([r['observation'] for r in records]),names=np.array(NAMES))
    pd.DataFrame(audit).to_csv(OUT/f'{cell}_audit.csv',index=False)
    report={'cell':cell,'workbooks':len(files),'discharge_rows':before,'duplicate_rows_removed':before-len(df),
            'segments':len(audit),'accepted':len(records),'excluded':pd.Series([r['excluded_reason'] for r in audit if r['excluded_reason']]).value_counts().to_dict(),
            'first_reference_capacity_ah':float(capacity[0]),'soh_min':float((capacity/capacity[0]).min()),
            'soh_max':float((capacity/capacity[0]).max()),'temperature_available':False,'sources':sources}
    (OUT/f'{cell}_manifest.json').write_text(json.dumps(report,indent=2))
    print(cell,'DONE',len(records),'of',len(audit),'excluded',report['excluded'],flush=True)
    return {k:v for k,v in report.items() if k!='sources'}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--cells',nargs='+',default=IDS,choices=IDS)
    args=parser.parse_args()
    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(prepare,args.cells))
    reports=[{k:v for k,v in json.loads((OUT/f'{cell}_manifest.json').read_text()).items() if k!='sources'}
             for cell in IDS if (OUT/f'{cell}_manifest.json').exists()]
    (OUT/'summary.json').write_text(json.dumps(reports,indent=2))
