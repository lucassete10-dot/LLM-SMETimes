"""Recompute accuracy and inspect every software handoff requirement."""
import json
from pathlib import Path
import shutil
import subprocess
import joblib
import numpy as np
import torch
from battery_smetimes.stm32_forecast import predict
from scripts.prepare_same_type_forecast import save_json, sha
from scripts.train_same_type_forecast import load_data


def main():
    torch.set_num_threads(2)
    run=Path(json.loads(Path('experiments/stm32_soh/latest.json').read_text())['folder'])
    package=Path('deployment/stm32_soh')/run.name
    audit=Path('experiments/soh_research_v5/audits/20260930T074750_792569Z')
    frozen=json.loads((audit/'config.json').read_text()); config=frozen['config']
    prepared=Path(json.loads((audit/'audit.json').read_text())['prepared_folder'])
    if sha(prepared/'manifest.json')!=frozen['prepared_manifest_sha256'] or sha(config['integrity_audit'])!=frozen['integrity_sha256']:
        raise ValueError('frozen data or quality rules changed')
    _,cells=load_data(prepared,config['integrity_audit'])
    registered=json.loads((audit/'cell_registry.json').read_text())['cells']
    for key,c in cells.items():
        c.update({f:registered[key][f] for f in ('physical_family','reference_established_source_cycle')})
    plans=json.loads((run/'splits.json').read_text())['plans']
    original=json.loads((audit/'splits.json').read_text())['plans']
    if plans!=[p for p in original if p['dataset'] in ('MIT','XJTU')]: raise ValueError('frozen folds changed')
    rows=[]; c_queries=0; max_resources={}; seen={'MIT':set(),'XJTU':set()}
    for plan in plans:
        folder=run/plan['name']; bundle=joblib.load(folder/'model.joblib')
        if set(bundle['train_ids'])!=set(plan['train']+plan['validation']): raise ValueError('refit IDs changed')
        if set(bundle['train_ids'])&set(plan['test']): raise ValueError('test leakage')
        if {cells[k]['physical_family'] for k in bundle['train_ids']+plan['test']}!={bundle['physical_family']}: raise ValueError('mixed family')
        if seen[plan['dataset']]&set(plan['test']): raise ValueError('duplicate test cell')
        seen[plan['dataset']].update(plan['test'])
        weights=torch.load(folder/'model.pt',weights_only=True)
        if any(not torch.equal(value,bundle['network'].state_dict()[name]) for name,value in weights.items()): raise ValueError('model copies differ')
        resources=json.loads((folder/'c_export/resources.json').read_text())
        if resources['parameters']>50000 or resources['constant_bytes']>128*1024 or resources['inference_stack_arrays_bytes']>32*1024:
            raise ValueError('model array budget exceeded')
        max_resources[plan['dataset']]=resources
        checks=json.loads((folder/'c_export/replay.json').read_text())
        if len(checks)!=len(plan['test']) or any(row['status']!='passed' for row in checks): raise ValueError('incomplete C replay')
        c_queries+=sum(row['queries'] for row in checks)
        for mode in ('float','int8'):
            report=json.loads((folder/mode/'metrics.json').read_text())
            if report['skipped'] or set(r['cell_id'] for r in report['by_cell'])!=set(plan['test']): raise ValueError('missing test coverage')
            with np.load(folder/mode/'predictions.npz') as arrays:
                for key in plan['test']:
                    prefix=f'{key}__100__'; truth=arrays[prefix+'truth']; predictions=arrays[prefix+'prediction']; future=arrays[prefix+'cycle']
                    expected_truth=cells[key]['soh'][np.isfinite(cells[key]['soh'])&(cells[key]['soh']>0)&(cells[key]['label_cycle']>100)]
                    if not np.array_equal(truth,expected_truth): raise ValueError('test targets changed')
                    mape=float(100*np.mean(np.abs(predictions-truth)/truth))
                    row=next(r for r in report['by_cell'] if r['cell_id']==key)
                    if abs(mape-row['metrics']['mape_percent'])>1e-8: raise ValueError('metric mismatch')
                    clean={k:v for k,v in cells[key].items() if k not in ('soh','label_cycle','unfiltered_soh')}
                    clean['cycle']=np.r_[clean['cycle'],101,200]
                    clean['raw']=np.vstack([clean['raw'],np.full((2,clean['raw'].shape[1]),np.inf)])
                    replay=predict(dict(bundle,quantized=mode=='int8'),clean,100,future)
                    current=predict(dict(bundle,quantized=mode=='int8'),cells[key],100,future)
                    if not np.array_equal(replay,current): raise ValueError(f'future/label-free replay mismatch {key}/{mode}')
                    if np.max(np.abs(current-predictions))>2e-6: raise ValueError(f'saved prediction mismatch {key}/{mode}')
                    rows.append({'dataset':plan['dataset'],'mode':mode,'cell_id':key,'mape_percent':mape})
    if {k:len(v) for k,v in seen.items()}!={'MIT':134,'XJTU':55}: raise ValueError('incomplete entire-cohort coverage')
    summary={}
    for domain in seen:
        summary[domain]={}
        for mode in ('float','int8'):
            value=float(np.mean([r['mape_percent'] for r in rows if r['dataset']==domain and r['mode']==mode]))
            if value>6: raise ValueError('accuracy target not passed')
            summary[domain][mode]=value
    ar=shutil.which('arm-none-eabi-ar')
    for core in ('cortex-m3','cortex-m4'):
        for domain in ('MIT','XJTU'):
            directory=package/'arm_objects'/core/domain
            subprocess.run([ar,'rcs',str(directory/f'libsoh_{domain.lower()}.a'),
                str(directory/'soh_model.o'),str(directory/'soh_features.o')],check=True)
    for file in ('pipeline_replay.json','onnx_and_cli_verification.json','arm_cross_compile.json'):
        if json.loads((package/file).read_text())['status']!='passed': raise ValueError(f'failed artifact {file}')
    hardware=subprocess.run(['STM32_Programmer_CLI','-l'],capture_output=True,text=True,check=True).stdout
    (package/'hardware_detection.txt').write_text(hardware,encoding='utf-8')
    if 'No ST-Link detected' not in hardware or 'Total number of serial ports available: 0' not in hardware:
        raise ValueError('hardware state changed; inspect connected board before completing fallback')
    requirements=[
        {'requirement':'frozen MIT/XJTU quality rules and whole-cell folds','status':'proved','evidence':'hash checks; plans compared to frozen audit; family/isolation checks'},
        {'requirement':'small numeric model <=40 inputs; validation selection; no test-driven exclusion','status':'proved','evidence':'training source/config, 40-input state dicts, complete 134/55 coverage and candidate validation histories'},
        {'requirement':'each dataset cell-macro MAPE <=6%, float and final quantized predictions','status':'proved','evidence':'all saved truth/prediction arrays recomputed; summary in this audit'},
        {'requirement':'observed signals only; no SOH/capacity/future information during inference','status':'proved','evidence':'all 189 cells, both modes: labels removed and actual future Inf rows appended, predictions exactly unchanged'},
        {'requirement':'saved weights, preprocessors, curves, by-cell metrics and reproducible new-cell command','status':'proved','evidence':'all ten fold artifacts; CLI examples; ONNX verification; inspected representative curves'},
        {'requirement':'quantization, portable C export and software resource validation','status':'proved','evidence':'INT8 calibration source; all folds host C replay; M3/M4 objects and archives; resources JSON'},
        {'requirement':'STM32 hardware-unavailable branch: complete software artifacts and explicit pending board items','status':'proved','evidence':'hardware_detection.txt plus handoff manifest pending list'},
        {'requirement':'actual board Flash/RAM/latency and real-battery demonstration','status':'pending_hardware_as_explicitly_allowed','evidence':'no connected programmer/serial device; no claims of board success'}]
    report={'status':'passed_software_handoff_hardware_unavailable','training_run':str(run.resolve()),
        'recomputed_cell_macro_mape_percent':summary,'c_model_replay_queries':c_queries,
        'all_cohort_future_poison_and_label_removal_passed':True,'requirements':requirements,
        'scope':'full requested training and deployment preparation; explicit hardware-unavailable fallback, board verification remains pending',
        'limitations':['single training seed public development CV','some end-of-life curvature remains imperfect despite mean MAPE target','ONNX ST target-version compatibility not checked; portable ARM C route compiled','array/object resources are not linked-board totals']}
    save_json(package/'completion_audit.json',report)
    (package/'INTEGRATION.md').write_text('''# Integrating the portable model

Use `arm_objects/<core>/<dataset>/libsoh_<dataset>.a` with the corresponding
model header and soh_features.h, or compile the C sources directly in CubeIDE.
The archive uses a soft-float ABI; ensure your project ABI matches, or rebuild
the sources with your board's own compiler flags. Link libm and libc. Do not use
fast-math because NaN signifies missing known-condition values. Application code
validates registered physical family and operating protocol before selecting a
model, rejects Inf, and streams increasing-cycle features with the declared
300s MIT or 120s XJTU acquisition protocol. ADC/raw-signal feature extraction
must be integrated and checked on the actual board.

`model_float.onnx` and `model_int8.onnx` include imputation/normalization and use
opset19. The INT8 interchange graph uses QDQ and floating bias/output. Target ST
tool support remains to be checked; the verified portable C path is independent
of that conversion. deployment_requirements.txt pins the optional ONNX exporter.

From the repository root, run either dataset's example/command.ps1 for label-free
future predictions. Training results reflect each of ten held-out fold models;
these two example archives are fixed fold1 artifacts and were not chosen by
test errors. They remain useful held-out demonstrations, not a claim that one
single fitted model independently predicted all 189 cells.

Pending hardware steps: select STM32, create matching startup/linker/project,
measure total image/stack/heap/ADC buffers and query/curve latency, replay sensor
features on board, and evaluate a real battery using future diagnostic capacity
as independent ground truth. Host and ARM-object checks cannot replace these.
''',encoding='utf-8')
    manifest=json.loads((package/'handoff_manifest.json').read_text())
    manifest['files_sha256']={str(p.relative_to(package)):sha(p) for p in package.rglob('*') if p.is_file() and p.name!='handoff_manifest.json'}
    save_json(package/'handoff_manifest.json',manifest)
    print(json.dumps({'status':report['status'],'mape':summary,'c_queries':c_queries}))


if __name__=='__main__': main()
