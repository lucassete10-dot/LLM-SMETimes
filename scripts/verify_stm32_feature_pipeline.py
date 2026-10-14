"""Package fixed fold-1 models and verify the streaming C feature frontend."""
import argparse
import ctypes
import json
from pathlib import Path
import shutil
import subprocess
import joblib
import numpy as np
from battery_smetimes.stm32_forecast import RAW_COLUMNS, compact_input, predict
from scripts.prepare_same_type_forecast import save_json
from scripts.train_same_type_forecast import load_data

HEADER = '''#ifndef SOH_FEATURES_H
#define SOH_FEATURES_H
typedef struct {
    float initial[5][7], recent[10][7];
    int recent_cycle[10], count, first_cycle, last_cycle, initial_count, reference_count, minimum_count;
} SohHistory;
void soh_history_init(SohHistory *s, int xjtu);
int soh_history_update(SohHistory *s, int source_cycle, const float raw_seven[7]);
int soh_history_query(const SohHistory *s, int future_cycle, const float conditions[5], float input[40]);
#endif
'''
SOURCE = '''#include <math.h>
#include <string.h>
#include "soh_features.h"
void soh_history_init(SohHistory *s, int xjtu) {
    memset(s,0,sizeof(*s)); s->reference_count=xjtu?5:1; s->minimum_count=xjtu?6:3;
}
int soh_history_update(SohHistory *s,int cycle,const float raw[7]) {
    if(cycle>100) return 0;
    if(cycle<=0 || (s->count && cycle<=s->last_cycle)) return -1;
    for(int j=0;j<7;j++) if(isinf(raw[j])) return -1;
    if(!s->count) s->first_cycle=cycle;
    if(s->initial_count<s->reference_count) memcpy(s->initial[s->initial_count++],raw,7*sizeof(float));
    int n=s->count<10?s->count:10;
    if(n==10) { memmove(s->recent,s->recent+1,9*7*sizeof(float)); memmove(s->recent_cycle,s->recent_cycle+1,9*sizeof(int)); n=9; }
    memcpy(s->recent[n],raw,7*sizeof(float)); s->recent_cycle[n]=cycle;
    s->last_cycle=cycle; s->count++; return 1;
}
static float median(float a[10],int n) {
    if(!n) return NAN;
    for(int i=1;i<n;i++) { float t=a[i]; int j=i-1; while(j>=0 && a[j]>t) { a[j+1]=a[j]; j--; } a[j+1]=t; }
    return n%2?a[n/2]:0.5f*(a[n/2-1]+a[n/2]);
}
int soh_history_query(const SohHistory *s,int future,const float conditions[5],float x[40]) {
    if(s->count<s->minimum_count || s->initial_count<s->reference_count || future<=100) return -1;
    int n=s->count<10?s->count:10;
    for(int j=0;j<7;j++) {
        float a[10]; int k=0;
        for(int i=0;i<s->initial_count;i++) if(isfinite(s->initial[i][j])) a[k++]=s->initial[i][j];
        float ref=median(a,k); k=0;
        for(int i=n>3?n-3:0;i<n;i++) if(isfinite(s->recent[i][j])) a[k++]=s->recent[i][j];
        float last=median(a,k); x[j]=last; x[7+j]=ref; x[14+j]=last-ref;
        /* Double accumulators agree with the Python trend calculation. */
        double ts=0,vs=0,num=0,den=0; int valid=0;
        for(int i=0;i<n;i++) if(isfinite(s->recent[i][j])) { ts+=s->recent_cycle[i]; vs+=s->recent[i][j]; valid++; }
        if(valid<2) { x[21+j]=NAN; continue; }
        ts/=valid; vs/=valid;
        for(int i=0;i<n;i++) if(isfinite(s->recent[i][j])) { double t=s->recent_cycle[i]-ts; num+=t*(s->recent[i][j]-vs); den+=t*t; }
        x[21+j]=den>0?(float)(100*num/den):NAN;
    }
    x[28]=1; x[29]=s->last_cycle/100.0f; x[30]=s->count/100.0f; x[31]=(s->last_cycle-s->first_cycle)/100.0f;
    for(int j=0;j<5;j++) x[32+j]=conditions[j];
    x[37]=future/100.0f; x[38]=(future-100)/100.0f; x[39]=log1pf((float)(future-100)); return 0;
}
'''


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run',type=Path)
    args=parser.parse_args()
    run=args.run or Path(json.loads(Path('experiments/stm32_soh/latest.json').read_text())['folder'])
    frozen=Path('experiments/soh_research_v5/audits/20260930T074750_792569Z')
    config=json.loads((frozen/'config.json').read_text())['config']
    data=Path(json.loads((frozen/'audit.json').read_text())['prepared_folder'])
    _,cells=load_data(data,config['integrity_audit'])
    registered=json.loads((frozen/'cell_registry.json').read_text())['cells']
    for key,c in cells.items():
        c.update({f:registered[key][f] for f in ('physical_family','reference_established_source_cycle')})
    plans=json.loads((run/'splits.json').read_text())['plans']
    output=Path('deployment/stm32_soh')/run.name
    output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,output/'verification_source.py')
    shutil.copy2('battery_smetimes/stm32_forecast.py',output/'stm32_forecast.py')
    header=output/'soh_features.h'; header.write_text(HEADER)
    source=output/'soh_features.c'; source.write_text(SOURCE)
    gcc=shutil.which('gcc')
    if not gcc: raise RuntimeError('gcc required for host frontend validation')
    subprocess.run([gcc,'-O2','-std=c99','-shared',str(source),'-o',str(output/'frontend.dll'),'-lm'],check=True)
    class History(ctypes.Structure):
        _fields_=[('initial',(ctypes.c_float*7)*5),('recent',(ctypes.c_float*7)*10),
                  ('recent_cycle',ctypes.c_int*10)]+[(f,ctypes.c_int) for f in ('count','first_cycle','last_cycle','initial_count','reference_count','minimum_count')]
    lib=ctypes.CDLL(str((output/'frontend.dll').resolve()))
    ptr=ctypes.POINTER(ctypes.c_float)
    lib.soh_history_init.argtypes=[ctypes.POINTER(History),ctypes.c_int]
    lib.soh_history_update.argtypes=[ctypes.POINTER(History),ctypes.c_int,ptr]
    lib.soh_history_query.argtypes=[ctypes.POINTER(History),ctypes.c_int,ptr,ptr]
    results=[]
    for domain in ('MIT','XJTU'):
        plan=next(p for p in plans if p['name']==f'{domain}_fold1')
        model_path=run/plan['name']; bundle=joblib.load(model_path/'model.joblib')
        dest=output/domain; dest.mkdir()
        for file in ('soh_model.c','soh_model.h','resources.json'):
            shutil.copy2(model_path/'c_export'/file,dest/file)
        for file in ('model.joblib','model.pt','model.json'):
            shutil.copy2(model_path/file,dest/file)
        subprocess.run([gcc,'-O2','-std=c99','-shared',str(dest/'soh_model.c'),'-o',str(dest/'model.dll'),'-lm'],check=True)
        net=ctypes.CDLL(str((dest/'model.dll').resolve())); net.soh_predict.argtypes=[ptr]; net.soh_predict.restype=ctypes.c_float
        for key in plan['test']:
            c=cells[key]; history=History(); lib.soh_history_init(ctypes.byref(history),domain=='XJTU')
            for cycle,row in zip(c['cycle'],c['raw']):
                seven=np.ascontiguousarray(row[RAW_COLUMNS[domain]],dtype=np.float32)
                if lib.soh_history_update(ctypes.byref(history),int(cycle),seven.ctypes.data_as(ptr))<0: raise ValueError('invalid history')
            future=c['label_cycle'][np.isfinite(c['soh'])&(c['soh']>0)&(c['label_cycle']>100)]
            expected_features=compact_input(c,100,future)
            conditions=np.ascontiguousarray(expected_features[0,32:37],dtype=np.float32)
            actual=[]; feature_error=0
            for i,cycle in enumerate(future):
                vector=np.zeros(40,dtype=np.float32)
                if lib.soh_history_query(ctypes.byref(history),int(cycle),conditions.ctypes.data_as(ptr),vector.ctypes.data_as(ptr)):
                    raise ValueError('C feature query failed')
                if not np.array_equal(np.isnan(vector),np.isnan(expected_features[i])): raise ValueError('missing-value mask mismatch')
                feature_error=max(feature_error,float(np.nanmax(np.abs(vector-expected_features[i]))))
                actual.append(net.soh_predict(vector.ctypes.data_as(ptr)))
            expected=predict(dict(bundle,quantized=True),c,100,future)
            error=float(np.max(np.abs(np.array(actual)-expected)))
            if error>1e-3 or feature_error>1e-4: raise ValueError(f'C frontend discrepancy {key}: {feature_error}, {error}')
            results.append({'dataset':domain,'cell_id':key,'queries':len(future),'max_feature_difference':feature_error,'max_soh_difference':error})
        from prepare_mit import ENHANCED_NAMES
        from prepare_xjtu import FEATURE_NAMES
        names=ENHANCED_NAMES if domain=='MIT' else FEATURE_NAMES
        save_json(dest/'input_contract.json',{'physical_family':bundle['physical_family'],'source_fold':plan['name'],
            'held_out_test_ids':plan['test'],'raw_seven_columns':[names[i] for i in RAW_COLUMNS[domain]],
            'raw_seven_indices':RAW_COLUMNS[domain],'feature_blocks':['last3 median (7)','first1 MIT/first5 XJTU median (7)','last-initial (7)','last10 least-squares slope per100 cycles (7)','origin,last,count,span divided100 (4)','charge_first_C,charge_last_C,charge_switch_fraction,diagnostic_discharge_C,cutoff_V (5)','future/100,horizon/100,log1p(horizon) (3)'],
            'observation_origin':100,'reference_count':1 if domain=='MIT' else 5,'streaming_history_bytes':ctypes.sizeof(History),
            'raw_signal_extraction':'compute seven named features using existing 300s MIT / 120s XJTU standardized observation protocol; frontend starts after this sensor processing',
            'evaluation':'fold1 weights selected only by validation; this deployment package covers fold1 holdout, full CV metrics refer to ten respective fold models'})
    save_json(output/'pipeline_replay.json',{'status':'passed','platform':'Windows host gcc, not STM32','cells':len(results),
        'queries':sum(r['queries'] for r in results),'max_soh_difference':max(r['max_soh_difference'] for r in results),'by_cell':results})
    (output/'README.md').write_text('''# STM32 SOH deployment preparation

New models: MIT/XJTU numeric 40-input ReLU MLP, symmetric INT8 weights and
activations, INT32 MACs, float scales/bias/output. Models are fixed fold1
validation-selected examples, not post-test selected winners.

Compile `soh_features.c` plus the chosen domain's `soh_model.c`; add both header
directories and link the math library. Use strict IEEE floating-point semantics:
do not enable `-ffast-math`, because missing-condition imputation uses NaN.
After checking physical family and known operating conditions, initialize the
history, stream increasing observed source-cycle feature rows (seven values in
the domain's input_contract.json order), query a future cycle >100, then call
`soh_predict`. Unknown condition entries are NaN and use training-only medians.
Do not pass infinity. Reject observations with incompatible sampling or type.
The streaming history plus per-query workspace has fixed memory, no heap and no
LLM/online connection. The seven raw features still require sensor preprocessing
matching the training protocol; this package does not configure ADC hardware.

`pipeline_replay.json` verifies history -> 40 features -> quantized C predictions
against Python on all fold1 unseen MIT/XJTU cells. All other folds' C verification
and predictions are retained under the training run. Actual STM32 Flash, RAM,
timing and ADC replay remain unmeasured pending a board model and connection.
''',encoding='utf-8')
    print(output.resolve()); print(json.dumps({'cells':len(results),'max_soh_difference':max(r['max_soh_difference'] for r in results)}))


if __name__=='__main__': main()
