"""Real raw-signal replay of the constant-scratch C feature extractor.

This reference batch API receives segmented stages; it does not validate ADC
segmentation, caller waveform storage, or board memory/latency.
"""
import ctypes
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import logging
import traceback
import h5py
import joblib
import numpy as np
from scipy.io import loadmat
from prepare_mit import discharge_slice,read_ref,feature_vector,GRID
from prepare_xjtu import CHANNELS,stage_features
from adapt_nasa_mat_v2 import prefix_features
from scripts.prepare_same_type_forecast import save_json,sha
from scripts.train_stm32_compact_trees import AUDIT
from scripts.train_stm32_compact_trees import predict
from scripts.train_same_type_forecast import load_data
from battery_smetimes.research_baselines import signal_prefix
from battery_smetimes.stm32_multiseed import predict as predict_mlp


class Moments(ctypes.Structure):
    _fields_=[('mean',ctypes.c_double),('m2',ctypes.c_double),('n',ctypes.c_size_t)]


class WaveStream(ctypes.Structure):
    _fields_=[('grid',(ctypes.c_double*11)*3),('previous',ctypes.c_double*4)]+[(name,ctypes.c_double) for name in ('end','step','seen_time','charge','energy','max_temperature')]+[(name,Moments) for name in ('voltage','current','absolute_current','temperature')]+[('count',ctypes.c_size_t)]+[(name,ctypes.c_int) for name in ('family','grid_count','next_grid','temperature_ok','have_previous','boundary_seen','error','finished')]

RUN_DIR=None

def main():
    global RUN_DIR
    parser=argparse.ArgumentParser();parser.add_argument('--package',type=Path);parser.add_argument('--app',type=Path,help='Linked reference app package; requires matching --package');parser.add_argument('--stream',action='store_true');parser.add_argument('--all',action='store_true',help='All cells; package replay includes train/validation cells and is numeric validation, not an accuracy test');args=parser.parse_args()
    if args.app and not args.package:parser.error('--app requires --package')
    frozen=json.loads((AUDIT/'config.json').read_text())
    prepared=Path(json.loads((AUDIT/'audit.json').read_text())['prepared_folder'])
    if sha(prepared/'manifest.json')!=frozen['prepared_manifest_sha256']:raise ValueError('changed data')
    manifest=json.loads((prepared/'manifest.json').read_text())
    records={r['cell_id']:r for r in manifest['cells']}
    plans=json.loads((AUDIT/'splits.json').read_text())['plans']
    _,cells=load_data(prepared,frozen['config']['integrity_audit'])
    registry=json.loads((AUDIT/'cell_registry.json').read_text())['cells']
    for key,c in cells.items():c.update({f:registry[key][f] for f in ('physical_family','reference_established_source_cycle')})
    out=Path('experiments/stm32_waveform_replay')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True)
    RUN_DIR=out
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'replay.log',encoding='utf-8'),logging.StreamHandler()])
    shutil.copy2(__file__,out/'verify_source.py')
    for name in ('soh_waveform.c','soh_waveform.h'):shutil.copy2(Path('deployment/common')/name,out/name)
    subprocess.run([shutil.which('gcc'),'-std=c99','-O2','-Wall','-Wextra','-Werror','-shared',str(out/'soh_waveform.c'),'-o',str(out/'waveform.dll'),'-lm'],check=True,capture_output=True,text=True)
    lib=ctypes.CDLL(str((out/'waveform.dll').resolve()));ptr=ctypes.POINTER(ctypes.c_double)
    lib.soh_waveform_features.argtypes=[ctypes.c_int,ctypes.c_size_t,ptr,ptr,ptr,ptr,ctypes.POINTER(ctypes.c_float)]
    lib.soh_wave_stream_init.argtypes=[ctypes.POINTER(WaveStream),ctypes.c_int]
    lib.soh_wave_stream_state_size.restype=ctypes.c_size_t
    if lib.soh_wave_stream_state_size()!=ctypes.sizeof(WaveStream):raise ValueError('C/Python streaming state ABI mismatch')
    lib.soh_wave_stream_push.argtypes=[ctypes.POINTER(WaveStream)]+[ctypes.c_double]*4
    lib.soh_wave_stream_finish.argtypes=[ctypes.POINTER(WaveStream),ctypes.POINTER(ctypes.c_float)]
    def extract(family,arrays):
        data=[np.ascontiguousarray(x,dtype=np.float64) for x in arrays]
        result=np.empty({1:51,2:44,3:23}[family],dtype=np.float32)
        status=lib.soh_waveform_features(family,len(data[0]),*[x.ctypes.data_as(ptr) for x in data],result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        if status:raise ValueError(f'C extractor rejected physical stage: {status}')
        if app is not None and current_cycle in app_visible:
            if app.soh_app_begin_stage(current_cycle,current_discharge)!=0:raise ValueError('app rejected valid stage event')
            for row in zip(*data):
                if app.soh_app_sample(*row)<0:raise ValueError('app rejected valid raw sample')
            if app.soh_app_end_stage()!=0:raise ValueError('app rejected complete raw stage')
        if args.stream:
            state=WaveStream();lib.soh_wave_stream_init(ctypes.byref(state),family)
            for row in zip(*data):
                if lib.soh_wave_stream_push(ctypes.byref(state),*row)<0:raise ValueError('stream rejected valid sample')
            online=np.empty_like(result)
            if lib.soh_wave_stream_finish(ctypes.byref(state),online.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))!=0:raise ValueError('stream finish failed')
            np.testing.assert_allclose(online,result,rtol=1e-6,atol=1e-5,equal_nan=True)
            result=online
        return result
    reports=[]
    cases=[(domain,key) for domain in ('MIT','XJTU','NASA') for key in (sorted(k for k,r in records.items() if r['metadata']['dataset']==domain) if args.all else next(p for p in plans if p['name']==domain+'_fold1')['test'][:1])]
    for domain,key in cases:
        r=records[key]
        app=None;current_cycle=0;current_discharge=1;app_visible=set(cells[key]['cycle'].tolist())
        if args.app:
            reference=args.app/domain;app_sources=out/domain
            if not app_sources.exists():
                app_sources.mkdir()
                for name in ('soh_tree.c','soh_tree.h','soh_stream.c','soh_stream.h','soh_selected_columns.h'):shutil.copy2(reference/name,app_sources/name)
                optional_model=['soh_model.c'] if (reference/'soh_model.c').exists() else []
                for name in optional_model:
                    shutil.copy2(reference/name,app_sources/name);shutil.copy2(reference/'soh_model.h',app_sources/'soh_model.h')
                for name in ('soh_waveform.c','soh_waveform.h','soh_reference_app.c','soh_reference_app.h'):shutil.copy2(Path('deployment/common')/name,app_sources/name)
                for name in ('soh_tree.c','soh_stream.c',*optional_model):
                    if sha(app_sources/name)!=sha(args.package/domain/name):raise ValueError('app and baseline model/frontend mismatch')
                subprocess.run([shutil.which('gcc'),'-std=c99','-O2','-Wall','-Wextra','-Werror','-DSOH_HOST_VERIFY','-shared',*[str(app_sources/n) for n in ('soh_tree.c','soh_stream.c','soh_waveform.c','soh_reference_app.c',*optional_model)],'-o',str(app_sources/'app.dll'),'-lm'],check=True,capture_output=True,text=True)
            app=ctypes.CDLL(str((app_sources/'app.dll').resolve()));app.soh_app_begin_stage.argtypes=[ctypes.c_int,ctypes.c_int]
            app.soh_app_sample.argtypes=[ctypes.c_double]*4
            app.soh_app_predict.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float)]
            app.soh_app_init()
            if app.soh_app_sample(0,4,-1,25)>=0 or app.soh_app_end_stage()>=0:raise ValueError('app accepted samples without active stage')
            if domain=='XJTU' and app.soh_app_begin_stage(1,1)>=0:raise ValueError('app accepted discharge without matching charge')
            if domain!='XJTU' and app.soh_app_begin_stage(1,0)>=0:raise ValueError('app accepted unsupported charge stage')
        with np.load(r['file'],allow_pickle=False) as z:
            expected_cycles=z['cycle'];frozen_raw=z['raw_features']
        produced={};source=Path(r['raw_source'])
        if domain=='MIT':
            with h5py.File(source,'r') as f:
                b=f['batch'];index=int(key.split('c')[1]);summary=f[b['summary'][index,0]];data=f[b['cycles'][index,0]]
                numbers=np.asarray(summary['cycle']).reshape(-1).astype(int)
                for j in np.flatnonzero(np.isin(numbers,expected_cycles)):
                    current_cycle=int(numbers[j]);current_discharge=1
                    t,v,i,T=[read_ref(f,data[name][j,0]) for name in ('t','V','I','T')]
                    run=discharge_slice(t,i,v)
                    if run is None:raise ValueError('frozen visible stage missing')
                    py=feature_vector(t[run],i[run],v[run],T[run],GRID)
                    seconds=np.maximum.accumulate((t[run]-t[run][0])*60)
                    value=extract(1,[seconds,v[run],i[run],T[run]])
                    np.testing.assert_allclose(value,py,rtol=1e-6,atol=1e-5,equal_nan=True);produced[int(numbers[j])]=value
        elif domain=='NASA':
            cycles=loadmat(source,variable_names=[key],simplify_cells=True)[key]['cycle'];count=0
            for cycle in cycles:
                if cycle['type']!='discharge':continue
                count+=1
                if count not in expected_cycles:continue
                current_cycle=count;current_discharge=1
                d=cycle['data'];value=extract(2,[d[name] for name in ('Time','Voltage_measured','Current_measured','Temperature_measured')])
                np.testing.assert_allclose(value,prefix_features(d),rtol=1e-6,atol=1e-5);produced[count]=value
        else:
            data=loadmat(source,variable_names=['data'],simplify_cells=True)['data']
            for number in expected_cycles:
                current_cycle=int(number)
                cycle=data[int(number)-1];values=[]
                for stage in (0,2):
                    current_discharge=int(stage==2)
                    time=np.asarray(cycle['relative_time_min']).reshape(-1);cuts=np.flatnonzero(time==0)
                    a=cuts[stage];b=cuts[stage+1] if stage+1<len(cuts) else len(time)
                    t=time[a:b]*60;keep=np.isfinite(t)&(t<=120);order=np.argsort(t[keep]);unique,ix=np.unique(t[keep][order],return_index=True)
                    channels=[np.asarray(cycle[name]).reshape(-1)[a:b][keep][order][ix] for name in CHANNELS]
                    value=extract(3,[unique,*channels]);np.testing.assert_allclose(value,stage_features(cycle,stage),rtol=1e-6,atol=1e-5);values.append(value)
                produced[int(number)]=np.concatenate(values)
        raw=np.stack([produced[int(n)] for n in expected_cycles])
        np.testing.assert_allclose(raw,frozen_raw,rtol=1e-6,atol=1e-5,equal_nan=True)
        np.savez_compressed(out/(key+'.npz'),cycle=expected_cycles,raw_c=raw,raw_python=frozen_raw)
        report=dict(dataset=domain,cell_id=key,cycles=len(raw),raw_source=str(source),max_absolute_feature_difference=float(np.nanmax(abs(raw-frozen_raw))))
        if args.package:
            dest=args.package/domain;bundle=joblib.load(dest/'model.joblib');contract=json.loads((dest/'input_contract.json').read_text())
            predictor=predict_mlp if contract.get('model_kind')=='int8_mlp' else predict
            if contract.get('model_kind')=='int8_mlp':bundle=dict(bundle,quantized=True)
            report['model_input_role']='development_training' if key in bundle['train_ids'] else 'not_used_in_this_model_training'
            cols=contract['raw_columns'];k=len(cols)
            class History(ctypes.Structure):
                _fields_=[('initial',(ctypes.c_float*k)*5),('recent',(ctypes.c_float*k)*10),('recent_cycle',ctypes.c_int*10)]+[(n,ctypes.c_int) for n in ('count','first_cycle','last_cycle','initial_count','reference_count','minimum_count')]
            pipeline=ctypes.CDLL(str((dest/'pipeline.dll').resolve()));fp=ctypes.POINTER(ctypes.c_float)
            pipeline.soh_stream_init.argtypes=[ctypes.POINTER(History),ctypes.c_int]
            pipeline.soh_stream_update.argtypes=[ctypes.POINTER(History),ctypes.c_int,fp]
            pipeline.soh_stream_query.argtypes=[ctypes.POINTER(History),ctypes.c_int,fp,fp]
            pipeline.soh_tree_predict.argtypes=[fp];pipeline.soh_tree_predict.restype=ctypes.c_float
            history=History();pipeline.soh_stream_init(ctypes.byref(history),int(domain=='XJTU'));cell=cells[key]
            for n in cell['cycle']:
                row=np.ascontiguousarray(produced[int(n)][cols],dtype=np.float32)
                if pipeline.soh_stream_update(ctypes.byref(history),int(n),row.ctypes.data_as(fp))!=1:raise ValueError('raw-derived C stream update failed')
            future=cell['label_cycle'][np.isfinite(cell['soh'])&(cell['soh']>0)&(cell['label_cycle']>100)]
            conditions=np.ascontiguousarray(signal_prefix(cell,100)[-5:],dtype=np.float32);pred=[]
            for n in future:
                query=np.empty(contract['full_input_width'],dtype=np.float32)
                if pipeline.soh_stream_query(ctypes.byref(history),int(n),conditions.ctypes.data_as(fp),query.ctypes.data_as(fp))!=0:raise ValueError('C query failed')
                pred.append(pipeline.soh_tree_predict(query.ctypes.data_as(fp)))
            error=float(np.max(abs(np.asarray(pred)-predictor(bundle,cell,100,future))))
            if error>1e-5:raise ValueError('raw-to-SOH pipeline diverged')
            report['raw_to_soh']=dict(status='passed',queries=len(future),max_absolute_soh_error=error,package=str(args.package.resolve()))
            if app is not None:
                actual=[]
                for n in future:
                    value=ctypes.c_float()
                    if app.soh_app_predict(int(n),conditions.ctypes.data_as(fp),ctypes.byref(value))!=0:raise ValueError('app prediction request rejected')
                    actual.append(value.value)
                error=float(np.max(abs(np.asarray(actual)-predictor(bundle,cell,100,future))))
                if error>1e-5:raise ValueError('integrated app and Python prediction differ')
                if app.soh_app_begin_stage(101,1)>=0:raise ValueError('app accepted future source cycle')
                value=ctypes.c_float()
                if app.soh_app_predict(100,conditions.ctypes.data_as(fp),ctypes.byref(value))>=0:raise ValueError('app accepted nonfuture query')
                invalid=conditions.copy();invalid[0]=np.inf
                if app.soh_app_predict(int(future[0]),invalid.ctypes.data_as(fp),ctypes.byref(value))>=0:raise ValueError('app accepted infinite conditions')
                report['integrated_app']=dict(status='passed',queries=len(future),max_absolute_soh_error=error,future_cycle_rejected=True,nonfuture_query_rejected=True,invalid_stage_sequence_rejected=True,infinite_conditions_rejected=True)
        reports.append(report)
        save_json(out/'progress.json',dict(complete=False,total=len(cases),reports=reports))
        logging.info('SAVED %s %s cycles=%s max_feature_difference=%.8g',domain,key,len(raw),report['max_absolute_feature_difference'])
    arm=shutil.which('arm-none-eabi-gcc');arm_result=None
    if arm:
        target=out/'waveform_m4.o'
        subprocess.run([arm,'-std=c99','-Os','-mcpu=cortex-m4','-mthumb','-mfloat-abi=soft','-fstack-usage','-Wall','-Wextra','-Werror','-c',str(out/'soh_waveform.c'),'-o',str(target)],check=True,capture_output=True,text=True)
        arm_result=dict(size=subprocess.run([shutil.which('arm-none-eabi-size'),str(target)],check=True,capture_output=True,text=True).stdout,stack=target.with_suffix('.su').read_text())
    save_json(out/'report.json',dict(status='passed',streaming=args.stream,all_cells=args.all,stream_state_bytes=ctypes.sizeof(WaveStream) if args.stream else None,reports=reports,arm=arm_result,files_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()},
        reference_app=str(args.app.resolve()) if args.app else None,
        scope='all frozen early stages of '+('all cells (package model also replays development training cells; numeric check only)' if args.all else 'one fixed example cell per dataset; model_input_role records training membership')+'; not sensor/board validation',pending=['actual ADC sensor replay','board-specific stage event integration','board memory/latency measurement']))
    save_json(out/'progress.json',dict(complete=True,total=len(cases),reports=reports))
    print('OUTPUT='+str(out.resolve()))


if __name__=='__main__':
    try:main()
    except BaseException:
        if RUN_DIR is not None:save_json(RUN_DIR/'failure.json',dict(traceback=traceback.format_exc()))
        raise
