"""Passive, hash-verified conversion into the same LLM forecasting contract.

Preserves all original cell roles. No fitted predictors or answer coefficients.
Each dataset remains an isolated registered cohort; missing I/T stay missing.
"""
import argparse,json,shutil
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np
from scripts.prepare_same_type_forecast import save_json
from scripts.audit_llm_curve_repair import fingerprint

HUST_CONFIG=Path('experiments/hust_length_smoke/20261008T015438_626064Z/config.json')
UCONN_CONFIG=Path('experiments/uconn_sparse_probe/20261008T073624_949393Z/config.json')
NCA_CONFIG=Path('experiments/zhu2022_nca_forecast/20261004T161453_449950Z/config.json')
UCONN_INDEX=Path('data/interim/uconn_causal_cycles/20261008T073156_095768Z/index.json')
NCA_INDEX=Path('data/interim/nca_curve_cells/20261005T031838_298841Z/index.json')


def read_npz(entry):
    path=Path(entry['path'])
    if fingerprint(path)!=entry['sha256']:raise ValueError('source cell hash drift: '+str(path))
    with np.load(path,allow_pickle=False) as data:return {k:data[k].copy() for k in data.files}


def load_prepared_external(settings):
    index_path=Path(settings['unified_external_index'])
    if fingerprint(index_path)!=settings['unified_external_index_sha256']:raise ValueError('external index drift')
    manifest=json.loads(index_path.read_text(encoding='utf-8'));root=index_path.parent
    registry=root/'registry.json'
    if fingerprint(registry)!=manifest['registry_sha256']:raise ValueError('external registry drift')
    cells={}
    for entry in manifest['cells']:
        data=read_npz(entry)
        cells[entry['cell_id']]={**data,'metadata':entry['metadata'],'physical_family':entry['physical_family'],
            'reference_established_source_cycle':entry['reference_established_source_cycle']}
    plan=next(p for p in manifest['plans'] if p['name']==settings['fold'])
    roles=[set(plan[r]) for r in ('train','validation','test')]
    if any(roles[i]&roles[j] for i in range(3) for j in range(i)):raise ValueError('external cell role overlap')
    if not set.union(*roles)<=set(cells):raise ValueError('missing requested external cells')
    family=cells[plan['train'][0]]['physical_family']
    if any(cells[k]['physical_family']!=family for k in set.union(*roles)):raise ValueError('external mixed-family routing')
    return manifest,cells,plan,registry,root


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);args=ap.parse_args()
    root=args.output or Path('data/interim/llm_external')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True,exist_ok=False);registry=json.loads(Path('configs/battery_physical_registry.json').read_text(encoding='utf-8'))
    inputs=[HUST_CONFIG,UCONN_CONFIG,NCA_CONFIG,UCONN_INDEX,NCA_INDEX]
    hust_cfg=json.loads(HUST_CONFIG.read_text(encoding='utf-8'));hust_index=Path(hust_cfg['source_index']);inputs.append(hust_index)
    if fingerprint(hust_index)!=hust_cfg['index_sha256']:raise ValueError('HUST index drift')
    datasets={};plans=[];limitations={};provenance={}
    hust=json.loads(hust_index.read_text(encoding='utf-8'));records=[]
    for entry in hust['cells']:
        a=read_npz(entry);raw=a['early_raw'].astype(np.float32);reference=float(a['truth'][0])
        meta=entry['metadata'];rates=[float(r['rate_in_C']) if r.get('rate_in_C') is not None else np.nan for r in meta['charge_protocol']]
        conditions=[30.,rates[0],rates[-1],meta.get('nominal_capacity_in_Ah',1.1),float(meta['discharge_protocol'][0].get('rate_in_C') or np.nan)]
        records.append((entry['cell_id'],entry['physical_family'],a['early_cycle'],raw,a['cycle'],a['truth']/reference,float(a['cycle'][0]),conditions,
            {'Voltage':list(range(11)),'Current':list(range(16,27)),'Temperature':[]},3,'initial comparable observed discharge capacity'))
    datasets['HUST']=records;plans.append({**{k:hust_cfg[k] for k in ('train','validation','test')},'name':'HUST_frozen_split','dataset':'HUST'})
    limitations['HUST']=hust['limitations']
    uconn=json.loads(UCONN_INDEX.read_text(encoding='utf-8'));cfg=json.loads(UCONN_CONFIG.read_text(encoding='utf-8'));records=[]
    for entry in uconn['cells']:
        a=read_npz(entry);q=a['capacity_ah'];reference=float(q[0]);raw=a['pulse_voltage'].astype(np.float32)
        records.append((entry['cell_id'],entry['physical_family'],a['source_cycle'],raw,a['source_cycle'],q/reference,float(a['source_cycle'][0]),
            [float(entry['group_id']),np.nan,np.nan,np.nan,np.nan],{'Voltage':list(range(raw.shape[1])),'Current':[],'Temperature':[]},1,'initial observed RPT discharge capacity'))
    datasets['UConn_ISU_ILCC']=records;plans.append({**{k:cfg[k] for k in ('train','validation','test')},'name':'UConn_frozen_split','dataset':'UConn_ISU_ILCC'})
    limitations['UConn_ISU_ILCC']=['Sparse measured pulse voltage only; no measured I/T in this processed release.','Protocol group is known, physical protocol mapping not independently verified.','Original 11 reserved test cells stay excluded from TRAIN/validation; no claim of certified independent confirmation.','Unknown aging intervals truncate source coordinates; no interpolation or future filling.']
    index=json.loads(NCA_INDEX.read_text(encoding='utf-8'));cfg=json.loads(NCA_CONFIG.read_text(encoding='utf-8'));records=[]
    for cid,entry in index['cells'].items():
        path=NCA_INDEX.parent/entry['file']
        if fingerprint(path)!=entry['sha256']:raise ValueError('NCA cell drift')
        a=joblib.load(path) # locally generated, pinned data-only bundles; no estimator prediction
        known=np.asarray(a['conditions'],float).ravel();conditions=list(known[:5])+[np.nan]*max(0,5-len(known))
        records.append((cid,a['physical_family'],a['early_cycle'],np.asarray(a['early_raw'],np.float32),a['cycle'],a['truth'],1.,conditions,
            {'Voltage':list(range(11)),'Current':list(range(16,27)),'Temperature':[]},3,'initial comparable discharge capacity; historical future label cache'))
    datasets['Zhu2022_NCA']=records;plans.extend({**p,'dataset':'Zhu2022_NCA'} for p in cfg['plans'])
    limitations['Zhu2022_NCA']=['Measured V/I in fixed first 900 seconds; no measured T.','Future labels in the historical cache start after cycle 100; no synthetic origin supervision.','Registered source cohort; separate family predictor, no cross-chemistry pooling.']
    all_entries=[]
    for dataset,rows in datasets.items():
        for cid,family,cycle,raw,x,y,ref_cycle,known,channels,minimum,reference in rows:
            if len(x)!=len(y) or np.any(np.diff(x)<=0) or raw.shape[0]!=len(cycle):raise ValueError('source alignment: '+cid)
            folder=root/dataset;folder.mkdir(exist_ok=True);path=folder/(cid.replace('#','_')+'.npz')
            np.savez_compressed(path,cycle=np.asarray(cycle,dtype=np.float32),raw=raw,label_cycle=np.asarray(x,dtype=np.float64),soh=np.asarray(y,dtype=np.float64))
            metadata={'dataset':dataset,'known_conditions':[float(v) if v is not None and np.isfinite(v) else None for v in known],'signal_schema':{'width':raw.shape[1],'minimum_observations':minimum,'text_channel_columns':channels},'label_reference':reference}
            all_entries.append({'cell_id':cid,'path':str(path.resolve()),'sha256':fingerprint(path),'metadata':metadata,'physical_family':family,'reference_established_source_cycle':ref_cycle})
            if family not in registry['families']:registry['families'][family]={'datasets':[dataset],'status':'isolated_registered_source_cohort_specification_limitations_retained','specifications':{},'limitations':limitations[dataset],'sources':[str(p.resolve()) for p in inputs]}
    save_json(root/'registry.json',registry)
    save_json(root/'index.json',{'cells':all_entries,'plans':plans,'registry_sha256':fingerprint(root/'registry.json'),
        'source_sha256':{str(p):fingerprint(p) for p in inputs},'limitations':limitations,'all_source_cells_retained':True,
        'label_capacity_not_in_model_input':True,'no_teacher_or_estimator_used':True,'independent_confirmation_certified':False,'goal_complete':False})
    shutil.copy2(__file__,root/'prepare_source.py')
    save_json(Path('data/interim/llm_external/latest.json'),{'folder':str(root.resolve()),'index_sha256':fingerprint(root/'index.json')})
    print(json.dumps({'folder':str(root.resolve()),'counts':{k:len(v) for k,v in datasets.items()}},indent=2),flush=True)


if __name__=='__main__':main()
