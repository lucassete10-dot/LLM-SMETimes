"""Label-free external routing and public-replay separation for release trees."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from battery_smetimes.physical_types import load_registry,match_external_specifications
from battery_smetimes.forecast import metadata_vector,type_key
from scripts.train_same_type_forecast import load_data
from scripts.train_stm32_compact_trees import predict
from scripts.prepare_same_type_forecast import sha


def fingerprint(cycle,raw):
    raw=np.array(raw,dtype='<f4',order='C',copy=True);raw[np.isnan(raw)]=np.nan
    raw[raw==0]=0.0
    return hashlib.sha256(np.asarray(cycle,dtype='<i8').tobytes()+raw.tobytes()).hexdigest()


def load_context(package):
    package=Path(package);frozen=package/'frozen_audit'
    cfg=json.loads((frozen/'config.json').read_text());prepared=Path(json.loads((frozen/'audit.json').read_text())['prepared_folder'])
    registry_path=frozen/'source_snapshot/battery_physical_registry.json'
    for file,digest in [(prepared/'manifest.json',cfg['prepared_manifest_sha256']),(registry_path,cfg['registry_sha256']),(Path(cfg['config']['integrity_audit']),cfg['integrity_sha256'])]:
        if sha(file)!=digest:raise ValueError('release data/type/quality contract changed')
    _,cells=load_data(prepared,cfg['config']['integrity_audit']);records=json.loads((frozen/'cell_registry.json').read_text())['cells']
    for key,c in cells.items():c.update({f:records[key][f] for f in ('physical_family','reference_established_source_cycle')})
    return dict(package=package,registry=load_registry(registry_path),cells=cells,
        fingerprints={fingerprint(c['cycle'],c['raw']):key for key,c in cells.items()})


def validate_signals(cycle,raw):
    cycle=np.asarray(cycle);raw=np.asarray(raw)
    if cycle.ndim!=1 or not 3<=len(cycle)<=100 or cycle.dtype.kind not in 'iu':raise ValueError('require 3..100 integer source-cycle observations')
    if (cycle<1).any() or (cycle>100).any() or np.any(np.diff(cycle.astype(np.int64))<=0):raise ValueError('source cycles must strictly increase within 1..100')
    if raw.ndim!=2 or raw.shape[0]!=len(cycle) or raw.dtype.kind not in 'fiu' or np.isinf(raw).any():raise ValueError('invalid signal feature matrix')
    with np.errstate(over='ignore'):converted=raw.astype(np.float32)
    if np.isinf(converted).any():raise ValueError('features overflow float32')
    return cycle,converted


def read_signals(path):
    with np.load(path,allow_pickle=False) as value:
        if set(value.files)!={'cycle','raw_features'}:raise ValueError('signals must contain only cycle/raw_features; labels, capacity and future-life fields are forbidden')
        cycle=value['cycle'];raw=value['raw_features']
    return validate_signals(cycle,raw)


def load_bundle(context,domain):
    path=context['package']/domain/'model.joblib'
    manifest=json.loads((context['package']/'manifest.json').read_text())
    hashes={name.replace('\\','/'):digest for name,digest in manifest['files_sha256'].items()}
    if sha(path)!=hashes.get(domain+'/model.joblib'):raise ValueError('release weights changed')
    return joblib.load(path)


def route(context,cycle,raw,request=None,public_cell=None):
    cycle,raw=validate_signals(cycle,raw)
    known=context['fingerprints'].get(fingerprint(cycle,raw));cells=context['cells']
    if public_cell:
        if request is not None or public_cell not in cells or known!=public_cell:raise ValueError('public replay needs exact audited signal fingerprint and no external metadata override')
        cell=dict(cells[public_cell]);cell={k:v for k,v in cell.items() if k not in ('soh','label_cycle','unfiltered_soh')}
        cell.update(cycle=cycle,raw=raw)
        return cell,dict(mode='public_replay',cell_id=public_cell,independent_new_cell=False,public_signal_match=known)
    if not isinstance(request,dict) or set(request)!={'cell_id','specifications','operating_conditions'}:raise ValueError('request requires only cell_id/specifications/operating_conditions; dataset/family labels and lifetime fields cannot route a new cell')
    if not isinstance(request['cell_id'],str) or not request['cell_id'].strip():raise ValueError('provide nonempty cell_id')
    if not isinstance(request['specifications'],dict):raise ValueError('specifications must be an object')
    if set(request['specifications'])!=set(context['registry']['external_required_fields']):raise ValueError('provide exactly the documented intrinsic specification fields; dataset/lifetime labels are not specifications')
    match=match_external_specifications(request['specifications'],context['registry'])
    if match['status']!='matched':raise ValueError('unsupported/unverified external type: '+json.dumps(match,ensure_ascii=False))
    family=match['physical_family'];registration=context['registry']['families'][family]
    if registration['status']!='model_and_specifications_documented':raise ValueError('unverified model identity; external routing disabled')
    domain=registration['datasets'][0]
    if domain!='MIT':raise ValueError('external feature/operating contract not implemented for this family')
    cond=request['operating_conditions'];required={'charge_rate_steps_c','charge_switch_soc','diagnostic_discharge_rate_c','ambient_temperature_c'}
    if not isinstance(cond,dict) or set(cond)!=required:raise ValueError('provide charge_rate_steps_c, charge_switch_soc (0..1), diagnostic_discharge_rate_c, ambient_temperature_c')
    rates=cond['charge_rate_steps_c']
    if not isinstance(rates,list) or len(rates)!=2:raise ValueError('two charge rates required')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in [*rates,cond['charge_switch_soc'],cond['diagnostic_discharge_rate_c'],cond['ambient_temperature_c']]):raise ValueError('conditions require numeric non-boolean values')
    first,last=map(float,rates);soc=float(cond['charge_switch_soc']);diag=float(cond['diagnostic_discharge_rate_c']);ambient=float(cond['ambient_temperature_c'])
    if not np.isfinite([first,last,soc,diag,ambient]).all() or min(first,last,diag)<=0 or not 0<soc<1:raise ValueError('invalid operating conditions')
    if abs(ambient-registration['conditions']['ambient_temperature_c'])>0.1:raise ValueError('ambient temperature outside documented training condition')
    metadata=dict(dataset=domain,group='external',charge_protocol=f'{first:.8f}C({100*soc:.8f}%)-{last:.8f}C',charge_rate_steps_c=[first,last],diagnostic_discharge_rate_c=diag)
    query_conditions=metadata_vector(metadata,[type_key(metadata)])[-5:]
    bundle=load_bundle(context,domain)
    supported=[metadata_vector(cells[k]['metadata'],[type_key(cells[k]['metadata'])])[-5:] for k in bundle['train_ids']]
    if not any(np.allclose(query_conditions,c,rtol=0,atol=1e-6,equal_nan=True) for c in supported):raise ValueError('charging/diagnostic protocol not covered by training; no silent extrapolation')
    if raw.shape!=(len(cycle),51) or not np.isfinite(raw[:,:22]).all():raise ValueError('require MIT51 features, finite voltage/current samples; temperatures may be missing')
    if len(cycle)<90 or cycle[-1]<95:raise ValueError('external N100 forecast requires at least90 usable observations and coverage through source cycle95; insufficient observed history')
    specifications=registration['specifications'];voltage=raw[:,:11];current=raw[:,11:22]
    if (voltage<specifications['rated_voltage_min_v']-.1).any() or (voltage>specifications['rated_voltage_max_v']+.1).any():raise ValueError('voltage/unit error outside model physical contract')
    # Source records have a step transition at t=0 and ~4 A at stable grid
    # points. Do not invent amps from C-rate * rated capacity (1.1 Ah).
    train_current=np.concatenate([np.median(abs(cells[k]['raw'][:,12:22]),axis=1) for k in bundle['train_ids']])
    lo,hi=float(train_current.min()),float(train_current.max());margin=.1*hi
    stable=np.median(abs(current[:,1:]),axis=1)
    if (current>=0).any() or (stable<lo-margin).any() or (stable>hi+margin).any():raise ValueError('stable discharge current outside training-observed envelope; check units/protocol')
    cell=dict(cycle=cycle,raw=raw,metadata=metadata,physical_family=family,reference_established_source_cycle=1)
    reused=known is not None or request['cell_id'] in cells
    return cell,dict(mode='external',cell_id=request['cell_id'],independent_new_cell=False if reused else None,eligible_for_independent_evaluation=not reused,
        independence_status='public/development input' if reused else 'external identity/provenance not yet independently verified',public_signal_match=known)


def forecast(context,cell,provenance,end_cycle,step=1):
    if isinstance(end_cycle,bool) or not isinstance(end_cycle,int) or not 100<end_cycle<=100000 or not isinstance(step,int) or isinstance(step,bool) or step<1:raise ValueError('request finite integer future range >100, ≤100000, positive step')
    domain=cell['metadata']['dataset'];folder=context['package']/domain;bundle=load_bundle(context,domain)
    future=np.arange(101,end_cycle+1,step)
    if future[-1]!=end_cycle:future=np.r_[future,end_cycle]
    prediction=predict(bundle,cell,100,future)
    support=json.loads((folder/'supported_horizon.json').read_text());maximum=support['largest_development_target_source_cycle'];warnings=[]
    if provenance['independent_new_cell'] is False:warnings.append('public/development input, not independent new-cell evidence')
    elif provenance['independent_new_cell'] is None:warnings.append('external cell identity and independent provenance still require confirmation')
    if future[-1]>maximum:warnings.append('requested cycles exceed development support; extrapolation unreliable, not a true lifetime estimate')
    return dict(**provenance,origin_source_cycle=100,physical_family=cell['physical_family'],future_source_cycles=future.tolist(),predicted_soh=prediction.tolist(),
        soh_definition='capacity / initial comparable capacity',known_condition_assumption='future charging/diagnostic/temperature policy remains unchanged',
        reference_rule='median_first5_comparable' if domain=='XJTU' else 'first_comparable',
        normalization='initial_comparable_capacity',supported_max_cycle=maximum,warnings=warnings,model_sha256=sha(folder/'model.joblib'),input_signal_sha256=fingerprint(cell['cycle'],cell['raw']))
