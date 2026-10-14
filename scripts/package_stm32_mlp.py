"""Frozen fold1/seed1 INT8 MLP examples with seven-signal streaming frontend."""
import argparse
import ctypes
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import joblib
import numpy as np
from battery_smetimes.stm32_multiseed import RAW_COLUMNS,compact_input,predict
from battery_smetimes.research_baselines import signal_prefix
from scripts.prepare_same_type_forecast import save_json,sha
from scripts.train_same_type_forecast import load_data
from scripts.verify_stm32_feature_pipeline import HEADER,SOURCE


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();run=args.run
    if not json.loads((run/'progress.json').read_text())['complete']:raise ValueError('full training incomplete')
    config=json.loads((run/'config.json').read_text());seed=config['seeds'][0]
    frozen=run/'frozen_audit';audit=json.loads((frozen/'audit.json').read_text());prepared=Path(audit['prepared_folder'])
    contract=json.loads((frozen/'config.json').read_text())
    if sha(prepared/'manifest.json')!=contract['prepared_manifest_sha256']:raise ValueError('data changed')
    _,cells=load_data(prepared,contract['config']['integrity_audit']);registry=json.loads((frozen/'cell_registry.json').read_text())['cells']
    for key,c in cells.items():c.update({f:registry[key][f] for f in ('physical_family','reference_established_source_cycle')})
    out=Path('deployment/stm32_compact_mlp')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True)
    shutil.copy2(__file__,out/'package_source.py');shutil.copy2(Path('scripts/verify_stm32_feature_pipeline.py'),out/'frontend_source.py');reports=[]
    plans=json.loads((run/'splits.json').read_text())['plans']
    for domain in ('MIT','XJTU','NASA'):
        plan=next(r for r in plans if r['name']==domain+'_fold1');model=run/plan['name']/str(seed)
        bundle=joblib.load(model/'model.joblib');bundle['quantized']=True
        dest=out/domain;dest.mkdir();shutil.copy2(model/'model.joblib',dest/'model.joblib')
        for name in ('soh_model.c','soh_model.h','resources.json'):shutil.copy2(model/'c_export'/name,dest/name)
        header=HEADER.replace('SOH_FEATURES_H','SOH_STREAM_H').replace('SohHistory','SohStream').replace('soh_history_','soh_stream_')
        header=header.replace('#define SOH_STREAM_H','#define SOH_STREAM_H\n#define SOH_RAW_COUNT 7\n#define SOH_INPUT_WIDTH 40')
        source=SOURCE.replace('soh_features.h','soh_stream.h').replace('SohHistory','SohStream').replace('soh_history_','soh_stream_')
        (dest/'soh_stream.h').write_text(header,encoding='utf-8');(dest/'soh_stream.c').write_text(source,encoding='utf-8')
        (dest/'soh_tree.h').write_text('#ifndef SOH_TREE_H\n#define SOH_TREE_H\n/* Compatibility entry; actual model is INT8 MLP, not a tree. */\nfloat soh_tree_predict(const float input[40]);\n#endif\n',encoding='utf-8')
        (dest/'soh_tree.c').write_text('#include "soh_model.h"\n#include "soh_tree.h"\nfloat soh_tree_predict(const float input[40]) { return soh_predict(input); }\n',encoding='utf-8')
        save_json(dest/'input_contract.json',dict(model_kind='int8_mlp',raw_columns=RAW_COLUMNS[domain],full_input_width=40,origin=100,physical_family=bundle['physical_family'],selection='fixed fold1 and first seed, not best outer-test model',scope='same operating conditions and known physical family; unknown specs not automatically classified'))
        subprocess.run([shutil.which('gcc'),'-std=c99','-O2','-Wall','-Wextra','-Werror','-shared',*[str(dest/n) for n in ('soh_tree.c','soh_stream.c','soh_model.c')],'-o',str(dest/'pipeline.dll'),'-lm'],check=True,capture_output=True,text=True)
        class History(ctypes.Structure):
            _fields_=[('initial',(ctypes.c_float*7)*5),('recent',(ctypes.c_float*7)*10),('recent_cycle',ctypes.c_int*10)]+[(n,ctypes.c_int) for n in ('count','first_cycle','last_cycle','initial_count','reference_count','minimum_count')]
        lib=ctypes.CDLL(str((dest/'pipeline.dll').resolve()));fp=ctypes.POINTER(ctypes.c_float)
        lib.soh_stream_init.argtypes=[ctypes.POINTER(History),ctypes.c_int];lib.soh_stream_update.argtypes=[ctypes.POINTER(History),ctypes.c_int,fp]
        lib.soh_stream_query.argtypes=[ctypes.POINTER(History),ctypes.c_int,fp,fp];lib.soh_tree_predict.argtypes=[fp];lib.soh_tree_predict.restype=ctypes.c_float
        checks=[]
        for key in plan['test']:
            cell=cells[key];history=History();lib.soh_stream_init(ctypes.byref(history),int(domain=='XJTU'))
            for cycle,row in zip(cell['cycle'],cell['raw']):
                raw=np.ascontiguousarray(row[RAW_COLUMNS[domain]],dtype=np.float32)
                if lib.soh_stream_update(ctypes.byref(history),int(cycle),raw.ctypes.data_as(fp))!=1:raise ValueError('update failed')
            future=cell['label_cycle'][np.isfinite(cell['soh'])&(cell['soh']>0)&(cell['label_cycle']>100)]
            expected=compact_input(cell,100,future);conditions=np.ascontiguousarray(signal_prefix(cell,100)[-5:],dtype=np.float32);actual=[]
            for j,n in enumerate(future):
                x=np.empty(40,dtype=np.float32)
                if lib.soh_stream_query(ctypes.byref(history),int(n),conditions.ctypes.data_as(fp),x.ctypes.data_as(fp))!=0:raise ValueError('query failed')
                np.testing.assert_allclose(x,expected[j],rtol=1e-6,atol=1e-5,equal_nan=True)
                actual.append(lib.soh_tree_predict(x.ctypes.data_as(fp)))
            error=float(np.max(abs(np.asarray(actual)-predict(bundle,cell,100,future))))
            if error>1e-5:raise ValueError('INT8 frontend+model differs')
            checks.append(dict(cell_id=key,queries=len(future),max_absolute_soh_error=error))
        save_json(dest/'pipeline_replay.json',dict(checks=checks,history_bytes=ctypes.sizeof(History),query_buffer_bytes=160))
        reports.append(dict(dataset=domain,checks=checks))
    save_json(out/'manifest.json',dict(run=str(run.resolve()),seed=seed,reports=reports,files_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},
        scope='fixed fold1 examples; host numerical validation, not board or independent new-cell validation'))
    print('OUTPUT='+str(out.resolve()))


if __name__=='__main__':main()
