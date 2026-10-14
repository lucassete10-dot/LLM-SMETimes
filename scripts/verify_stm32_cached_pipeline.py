"""All-cell replay from hash-verified raw-C features to streaming C predictions.

Upstream sample-stream correctness is established by a completed waveform run;
this check avoids reloading every original MAT when model/selected columns change.
It is numerical composition evidence, not new-cell generalization or board proof.
"""
import argparse
import ctypes
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
import torch
from battery_smetimes.forecast import query_matrix
from battery_smetimes.research_baselines import signal_prefix
from battery_smetimes.stm32_multiseed import compact_input,predict as predict_mlp
from scripts.train_stm32_compact_trees import AUDIT,predict
from scripts.train_same_type_forecast import load_data
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--package',type=Path,required=True);parser.add_argument('--waveform-run',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(2)
    upstream=json.loads((args.waveform_run/'report.json').read_text())
    if upstream['status']!='passed' or not upstream['all_cells'] or not upstream['streaming']:raise ValueError('need completed full streaming waveform proof')
    frozen=json.loads((AUDIT/'config.json').read_text());prepared=Path(json.loads((AUDIT/'audit.json').read_text())['prepared_folder'])
    if sha(prepared/'manifest.json')!=frozen['prepared_manifest_sha256']:raise ValueError('data changed')
    _,cells=load_data(prepared,frozen['config']['integrity_audit']);records=json.loads((AUDIT/'cell_registry.json').read_text())['cells']
    for key,c in cells.items():c.update({f:records[key][f] for f in ('physical_family','reference_established_source_cycle')})
    if {r['cell_id'] for r in upstream['reports']}!=set(cells):raise ValueError('upstream cell coverage mismatch')
    out=Path('experiments/stm32_cached_pipeline')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True);shutil.copy2(__file__,out/'source.py');checks=[]
    for domain in ('MIT','XJTU','NASA'):
        dest=args.package/domain;contract=json.loads((dest/'input_contract.json').read_text());bundle=joblib.load(dest/'model.joblib')
        neural=contract.get('model_kind')=='int8_mlp';predictor=predict_mlp if neural else predict
        if neural:bundle=dict(bundle,quantized=True)
        cols=contract['raw_columns'];k=len(cols);width=contract['full_input_width']
        class History(ctypes.Structure):
            _fields_=[('initial',(ctypes.c_float*k)*5),('recent',(ctypes.c_float*k)*10),('recent_cycle',ctypes.c_int*10)]+[(n,ctypes.c_int) for n in ('count','first_cycle','last_cycle','initial_count','reference_count','minimum_count')]
        lib=ctypes.CDLL(str((dest/'pipeline.dll').resolve()));fp=ctypes.POINTER(ctypes.c_float)
        lib.soh_stream_init.argtypes=[ctypes.POINTER(History),ctypes.c_int];lib.soh_stream_update.argtypes=[ctypes.POINTER(History),ctypes.c_int,fp]
        lib.soh_stream_query.argtypes=[ctypes.POINTER(History),ctypes.c_int,fp,fp];lib.soh_tree_predict.argtypes=[fp];lib.soh_tree_predict.restype=ctypes.c_float
        for key,c in sorted(cells.items()):
            if c['metadata']['dataset']!=domain:continue
            file=args.waveform_run/(key+'.npz')
            if sha(file)!=upstream['files_sha256'][file.name]:raise ValueError('cached raw-C features changed')
            with np.load(file,allow_pickle=False) as z:cached=dict(zip(z['cycle'].tolist(),z['raw_c']))
            history=History();lib.soh_stream_init(ctypes.byref(history),int(domain=='XJTU'))
            for cycle,row in zip(c['cycle'],c['raw']):
                np.testing.assert_allclose(cached[int(cycle)],row,rtol=1e-6,atol=1e-5,equal_nan=True)
                raw=np.ascontiguousarray(cached[int(cycle)][cols],dtype=np.float32)
                if lib.soh_stream_update(ctypes.byref(history),int(cycle),raw.ctypes.data_as(fp))!=1:raise ValueError('update failed')
            future=c['label_cycle'][np.isfinite(c['soh'])&(c['soh']>0)&(c['label_cycle']>100)]
            prefix=signal_prefix(c,100);conditions=np.ascontiguousarray(prefix[-5:],dtype=np.float32)
            expected=compact_input(c,100,future) if neural else query_matrix(prefix,100,future);used=list(range(40)) if neural else bundle['columns'];actual=[]
            for j,n in enumerate(future):
                x=np.empty(width,dtype=np.float32)
                if lib.soh_stream_query(ctypes.byref(history),int(n),conditions.ctypes.data_as(fp),x.ctypes.data_as(fp))!=0:raise ValueError('query failed')
                np.testing.assert_allclose(x[used],expected[j,used],rtol=1e-6,atol=1e-5,equal_nan=True)
                actual.append(lib.soh_tree_predict(x.ctypes.data_as(fp)))
            error=float(np.max(abs(np.asarray(actual)-predictor(bundle,c,100,future))))
            if error>1e-5:raise ValueError(f'C mismatch {key}: {error}')
            checks.append(dict(dataset=domain,cell_id=key,queries=len(future),max_absolute_soh_error=error,raw_cache_sha256=sha(file),model_input_role='development_training' if key in bundle['train_ids'] else 'not_used_in_this_model_training'))
            save_json(out/'progress.json',dict(complete=False,checks=checks))
    save_json(out/'report.json',dict(status='passed',checks=checks,package=str(args.package.resolve()),package_manifest_sha256=sha(args.package/'manifest.json'),waveform_run=str(args.waveform_run.resolve()),waveform_report_sha256=sha(args.waveform_run/'report.json'),
        scope='hash-verified upstream raw-C features -> C cycle streaming frontend -> C model; includes development cells, numerical composition only, not stage-event app or board execution'))
    save_json(out/'progress.json',dict(complete=True,checks=checks));print('OUTPUT='+str(out.resolve()))


if __name__=='__main__':main()
