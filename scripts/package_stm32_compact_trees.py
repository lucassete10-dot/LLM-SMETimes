"""Generic streaming feature frontend, executable replay and ARM objects."""
import argparse
import ctypes
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import joblib
import numpy as np
from battery_smetimes.forecast import query_matrix
from battery_smetimes.research_baselines import signal_prefix
from scripts.prepare_same_type_forecast import save_json, sha
from scripts.train_same_type_forecast import load_data
from scripts.train_stm32_compact_trees import AUDIT, predict


def frontend(dest,bundle):
    width=(bundle['input_width']-12)//4
    raw_cols=sorted({i%width for i in bundle['columns'] if i<4*width});k=len(raw_cols)
    header=f'''#ifndef SOH_STREAM_H
#define SOH_STREAM_H
#define SOH_RAW_COUNT {k}
#define SOH_INPUT_WIDTH {bundle['input_width']}
typedef struct {{ float initial[5][SOH_RAW_COUNT],recent[10][SOH_RAW_COUNT];
 int recent_cycle[10],count,first_cycle,last_cycle,initial_count,reference_count,minimum_count; }} SohStream;
void soh_stream_init(SohStream *s,int xjtu);
int soh_stream_update(SohStream *s,int cycle,const float raw[SOH_RAW_COUNT]);
int soh_stream_query(const SohStream *s,int future,const float conditions[5],float input[SOH_INPUT_WIDTH]);
#endif
'''
    source='''#include <math.h>
#include <string.h>
#include "soh_stream.h"
static const unsigned raw_columns[SOH_RAW_COUNT]={COLUMNS};
void soh_stream_init(SohStream *s,int xjtu) { memset(s,0,sizeof(*s));s->reference_count=xjtu?5:1;s->minimum_count=xjtu?6:3; }
int soh_stream_update(SohStream *s,int cycle,const float raw[SOH_RAW_COUNT]) {
 if(cycle>100) return 0;
 if(cycle<=0||(s->count&&cycle<=s->last_cycle)) return -1;
 for(unsigned j=0;j<SOH_RAW_COUNT;j++) if(isinf(raw[j])) return -1;
 if(!s->count) s->first_cycle=cycle;
 if(s->initial_count<s->reference_count) memcpy(s->initial[s->initial_count++],raw,SOH_RAW_COUNT*sizeof(float));
 int n=s->count<10?s->count:10;
 if(n==10) { memmove(s->recent,s->recent+1,9*SOH_RAW_COUNT*sizeof(float));memmove(s->recent_cycle,s->recent_cycle+1,9*sizeof(int));n=9; }
 memcpy(s->recent[n],raw,SOH_RAW_COUNT*sizeof(float));s->recent_cycle[n]=cycle;s->last_cycle=cycle;s->count++;return 1;
}
static float median(float a[10],int n) {
 if(!n) return NAN;
 for(int i=1;i<n;i++) { float t=a[i];int j=i-1;while(j>=0&&a[j]>t) {a[j+1]=a[j];j--;}a[j+1]=t; }
 return n%2?a[n/2]:0.5f*(a[n/2-1]+a[n/2]);
}
int soh_stream_query(const SohStream *s,int future,const float conditions[5],float x[SOH_INPUT_WIDTH]) {
 if(s->count<s->minimum_count||s->initial_count<s->reference_count||future<=100) return -1;
 for(unsigned i=0;i<SOH_INPUT_WIDTH;i++) x[i]=NAN;
 int n=s->count<10?s->count:10;
 for(unsigned j=0;j<SOH_RAW_COUNT;j++) {
  float a[10];int m=0;unsigned col=raw_columns[j];
  for(int i=0;i<s->initial_count;i++) if(isfinite(s->initial[i][j])) a[m++]=s->initial[i][j];
  float initial=median(a,m);m=0;
  for(int i=n>3?n-3:0;i<n;i++) if(isfinite(s->recent[i][j])) a[m++]=s->recent[i][j];
  float last=median(a,m);x[col]=last;x[WIDTH+col]=initial;x[2*WIDTH+col]=last-initial;
  double ts=0,vs=0,num=0,den=0;int valid=0;
  for(int i=0;i<n;i++) if(isfinite(s->recent[i][j])) {ts+=s->recent_cycle[i];vs+=s->recent[i][j];valid++;}
  if(valid<2) continue;
  ts/=valid;vs/=valid;
  for(int i=0;i<n;i++) if(isfinite(s->recent[i][j])) {double t=s->recent_cycle[i]-ts;num+=t*(s->recent[i][j]-vs);den+=t*t;}
  x[3*WIDTH+col]=den>0?(float)(100*num/den):NAN;
 }
 x[4*WIDTH]=1;x[4*WIDTH+1]=s->last_cycle/100.0f;x[4*WIDTH+2]=s->count/100.0f;x[4*WIDTH+3]=(s->last_cycle-s->first_cycle)/100.0f;
 for(int i=0;i<5;i++) { if(isinf(conditions[i])) return -1;x[4*WIDTH+4+i]=conditions[i]; }
 x[4*WIDTH+9]=future/100.0f;x[4*WIDTH+10]=(future-100)/100.0f;x[4*WIDTH+11]=log1pf((float)(future-100));return 0;
}
'''.replace('COLUMNS',','.join(map(str,raw_cols))).replace('WIDTH',str(width))
    # Replace numeric primitive-width markers without changing INPUT_WIDTH.
    source=source.replace('SOH_INPUT_'+str(width),'SOH_INPUT_WIDTH')
    (dest/'soh_stream.h').write_text(header,encoding='utf-8');(dest/'soh_stream.c').write_text(source,encoding='utf-8')
    save_json(dest/'input_contract.json',dict(raw_columns=raw_cols,raw_schema_width=width,full_input_width=bundle['input_width'],
         primitive_count=k,origin=100,raw_input='selected per-cycle V/I/T statistics from the existing dataset-specific waveform extractor',
         raw_waveform_extractor_on_board='pending implementation and sensor replay',physical_family=bundle['physical_family']))
    return raw_cols


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    run=args.run;config=json.loads((run/'config.json').read_text());state=json.loads((run/'progress.json').read_text())
    completed={(r['fold'],r['seed']) for r in state['completed']};seed=config['seeds'][0]
    plans=json.loads((run/'splits.json').read_text())['plans']
    if not args.smoke and not state['complete']:raise ValueError('formal package requires completed training')
    candidates=[p for p in plans if p['name'].endswith('fold1') and (p['name'],seed) in completed]
    out=Path('deployment/stm32_compact_trees')/(run.name+'_'+datetime.now(timezone.utc).strftime('%H%M%S_%f'));out.mkdir(parents=True)
    shutil.copy2(Path(__file__),out/'package_source.py')
    frozen=json.loads((AUDIT/'config.json').read_text());prepared=Path(json.loads((AUDIT/'audit.json').read_text())['prepared_folder'])
    if sha(prepared/'manifest.json')!=frozen['prepared_manifest_sha256']:raise ValueError('data changed')
    _,cells=load_data(prepared,frozen['config']['integrity_audit'])
    records=json.loads((AUDIT/'cell_registry.json').read_text())['cells']
    for key,c in cells.items():c.update({f:records[key][f] for f in ('physical_family','reference_established_source_cycle')})
    reports=[];gcc=shutil.which('gcc');arm=shutil.which('arm-none-eabi-gcc');size=shutil.which('arm-none-eabi-size')
    for plan in candidates:
        model=run/plan['name']/str(seed);bundle=joblib.load(model/'model.joblib')
        dest=out/plan['dataset'];dest.mkdir()
        for name in ('soh_tree.c','soh_tree.h','resources.json'):shutil.copy2(model/'c_export'/name,dest/name)
        shutil.copy2(model/'model.joblib',dest/'model.joblib')
        raw_cols=frontend(dest,bundle);k=len(raw_cols)
        subprocess.run([gcc,'-std=c99','-O2','-Wall','-Wextra','-Werror','-shared',str(dest/'soh_tree.c'),str(dest/'soh_stream.c'),'-o',str(dest/'pipeline.dll'),'-lm'],check=True,capture_output=True,text=True)
        class History(ctypes.Structure):
            _fields_=[('initial',(ctypes.c_float*k)*5),('recent',(ctypes.c_float*k)*10),('recent_cycle',ctypes.c_int*10)]+[(n,ctypes.c_int) for n in ('count','first_cycle','last_cycle','initial_count','reference_count','minimum_count')]
        lib=ctypes.CDLL(str((dest/'pipeline.dll').resolve()));ptr=ctypes.POINTER(ctypes.c_float)
        lib.soh_stream_init.argtypes=[ctypes.POINTER(History),ctypes.c_int]
        lib.soh_stream_update.argtypes=[ctypes.POINTER(History),ctypes.c_int,ptr]
        lib.soh_stream_query.argtypes=[ctypes.POINTER(History),ctypes.c_int,ptr,ptr]
        lib.soh_tree_predict.argtypes=[ptr];lib.soh_tree_predict.restype=ctypes.c_float
        checks=[]
        for key in plan['test']:
            cell=cells[key];history=History();lib.soh_stream_init(ctypes.byref(history),int(plan['dataset']=='XJTU'))
            for cycle,row in zip(cell['cycle'],cell['raw']):
                if cycle>100:continue
                raw=np.ascontiguousarray(row[raw_cols],dtype=np.float32)
                if lib.soh_stream_update(ctypes.byref(history),int(cycle),raw.ctypes.data_as(ptr))!=1:raise ValueError('C stream update rejected visible cycle')
            future=cell['label_cycle'][np.isfinite(cell['soh'])&(cell['soh']>0)&(cell['label_cycle']>100)]
            prefix=signal_prefix(cell,100);conditions=np.ascontiguousarray(prefix[-5:],dtype=np.float32)
            expected_inputs=query_matrix(prefix,100,future);predictions=[];max_input_error=0
            for index,n in enumerate(future):
                value=np.empty(bundle['input_width'],dtype=np.float32)
                if lib.soh_stream_query(ctypes.byref(history),int(n),conditions.ctypes.data_as(ptr),value.ctypes.data_as(ptr))!=0:raise ValueError('C query failed')
                cols=bundle['columns'];np.testing.assert_allclose(value[cols],expected_inputs[index,cols],rtol=1e-6,atol=1e-5,equal_nan=True)
                diff=np.abs(value[cols]-expected_inputs[index,cols]);max_input_error=max(max_input_error,float(np.nanmax(diff)))
                predictions.append(lib.soh_tree_predict(value.ctypes.data_as(ptr)))
            error=float(np.max(abs(np.asarray(predictions)-predict(bundle,cell,100,future))))
            if error>1e-5:raise ValueError(f'C full pipeline prediction mismatch {key} {error}')
            checks.append(dict(cell_id=key,queries=len(future),max_input_error=max_input_error,max_prediction_error=error))
        objects=[]
        if arm and size:
            arm_dir=dest/'arm_cortex_m4';arm_dir.mkdir()
            for name in ('soh_tree','soh_stream'):
                target=arm_dir/(name+'.o')
                subprocess.run([arm,'-std=c99','-Os','-mcpu=cortex-m4','-mthumb','-mfloat-abi=soft','-ffunction-sections','-fdata-sections','-fstack-usage','-Wall','-Wextra','-Werror','-c',str(dest/(name+'.c')),'-o',str(target)],check=True,capture_output=True,text=True)
                objects.append(dict(source=name,size=subprocess.run([size,str(target)],check=True,capture_output=True,text=True).stdout,stack=target.with_suffix('.su').read_text()))
        report=dict(dataset=plan['dataset'],fold=plan['name'],seed=seed,checks=checks,arm_objects=objects,
                    history_bytes=ctypes.sizeof(History),full_input_work_buffer_bytes=bundle['input_width']*4,
                    status='host_pipeline_passed',scope='fixed fold1 example; no board, raw sensor feature extraction or linked firmware validation')
        save_json(dest/'pipeline_replay.json',report);reports.append(report)
    save_json(out/'manifest.json',dict(run=str(run.resolve()),smoke=args.smoke,reports=reports,
         files_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},
         pending=['raw waveform feature extraction','five-seed MLP comparison','linked firmware budget','board Flash/RAM/latency','independent new-battery evaluation']))
    print('OUTPUT='+str(out.resolve()),flush=True)


if __name__=='__main__':main()
