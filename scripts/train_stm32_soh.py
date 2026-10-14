"""Train new compact SOH models on frozen whole-cell folds, with C export."""
import argparse
import copy
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import joblib
import numpy as np
import torch
from battery_smetimes.stm32_forecast import RAW_COLUMNS, Scale, TinySOH, compact_input, pairs, predict, quantize
from battery_smetimes.research_baselines import evaluate_baseline
from scripts.prepare_same_type_forecast import save_json, sha
from scripts.train_same_type_forecast import load_data
from scripts.train_soh_research_v5 import save_scores


def train(x, y, weights, hidden, seed, epochs, validation=None, device='cuda'):
    torch.manual_seed(seed)
    net = TinySOH(hidden).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.001)
    tx, ty, tw = [torch.tensor(a, dtype=torch.float32, device=device) for a in (x, y, weights)]
    val = [(torch.tensor(a, dtype=torch.float32, device=device), b) for a, b in validation] if validation else []
    best, best_epoch, best_score, stale = None, epochs, float('inf'), 0
    history = []
    for epoch in range(1, epochs + 1):
        net.train()
        prediction = net(tx)
        loss = ((prediction - ty).square() * tw).sum()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        opt.step()
        if epoch % 5 == 0 or epoch == epochs:
            row = {'epoch': epoch, 'training_mse': float(loss.detach().cpu())}
            if val:
                net.eval()
                with torch.no_grad():
                    score = float(np.mean([100 * np.mean(np.abs(net(a).cpu().numpy() - b) / b) for a, b in val]))
                row['validation_cell_macro_mape_percent'] = score
                if score < best_score - 1e-5:
                    best_score, best_epoch, best, stale = score, epoch, copy.deepcopy(net.state_dict()), 0
                else:
                    stale += 1
            history.append(row)
            if val and stale >= 16:
                break
    if best is not None:
        net.load_state_dict(best)
    return net.cpu().eval(), best_epoch, best_score if val else None, history


def float_array(name, a):
    numbers = []
    for value in np.asarray(a).ravel():
        word = f'{float(value):.9g}'
        if '.' not in word and 'e' not in word:
            word += '.0'
        numbers.append(word + 'f')
    return f'static const float {name}[{len(numbers)}] = {{' + ','.join(numbers) + '};\n'


def export_c(folder, bundle):
    folder.mkdir()
    layers = bundle['int8_layers']
    text = '#include <math.h>\n#include <stdint.h>\n#include "soh_model.h"\n'
    for field in ('fill', 'mean', 'std'):
        text += float_array(field, getattr(bundle['scale'], field))
    for index, layer in enumerate(layers):
        weight = layer['weight'].ravel()
        text += f'static const int8_t w{index}[{len(weight)}] = {{' + ','.join(map(str, weight)) + '};\n'
        text += float_array(f's{index}', layer['weight_scale'])
        text += float_array(f'b{index}', layer['bias'])
    maximum = max(40, *bundle['hidden'])
    text += 'static int8_t quant(float x, float s) { float q=nearbyintf(x/s); return (int8_t)fmaxf(-127.0f,fminf(127.0f,q)); }\n'
    text += f'float soh_predict(const float input[40]) {{ float a[{maximum}],z[{maximum}]; int8_t q[{maximum}];\n'
    text += 'for(int i=0;i<40;i++) { float v=isnan(input[i])?fill[i]:input[i]; a[i]=fmaxf(-8.0f,fminf(8.0f,(v-mean[i])/std[i])); }\n'
    for index, layer in enumerate(layers):
        outputs, inputs = layer['weight'].shape
        scale = float_array('temp', [layer['input_scale']]).split('{')[1].split('}')[0]
        text += f'for(int i=0;i<{inputs};i++) q[i]=quant(a[i],{scale});\n'
        text += f'for(int j=0;j<{outputs};j++) {{ int32_t acc=0; for(int i=0;i<{inputs};i++) acc+=(int32_t)q[i]*w{index}[j*{inputs}+i]; z[j]=(float)acc*{scale}*s{index}[j]+b{index}[j]; }}\n'
        text += f'for(int j=0;j<{outputs};j++) a[j]=' + ('fmaxf(0.0f,z[j]);\n' if index < 2 else 'z[j];\n')
    text += 'return 1.0f+a[0]; }\n'
    (folder / 'soh_model.c').write_text(text)
    (folder / 'soh_model.h').write_text('#ifndef SOH_MODEL_H\n#define SOH_MODEL_H\n/* 40 compact unscaled features; caller checks type, finite values and observation availability. */\nfloat soh_predict(const float input[40]);\n#endif\n')
    const_bytes = sum(layer['weight'].nbytes + layer['weight_scale'].nbytes + layer['bias'].nbytes + 4 for layer in layers) + 3 * 40 * 4
    save_json(folder / 'resources.json', {'parameters':sum(p.numel() for p in bundle['network'].parameters()),
        'constant_bytes':const_bytes,'inference_stack_arrays_bytes':maximum * 9,
        'arithmetic':'INT8 weights and activations, INT32 dot products, FP32 scales/bias/output',
        'board_flash_ram_latency':'not measured: no specified connected STM32 board'})


def c_replay(folder, bundle, raw):
    import ctypes
    gcc = shutil.which('gcc')
    if not gcc:
        return {'status':'compiler unavailable'}
    # Windows locks a loaded DLL. Compile once per unique fold directory.
    if not (folder/'soh_model.dll').exists():
        result=subprocess.run([gcc, '-O2','-std=c99','-shared',str(folder/'soh_model.c'),'-o',str(folder/'soh_model.dll'),'-lm'],capture_output=True,text=True)
        if result.returncode:
            raise RuntimeError('C compilation failed: '+result.stderr)
    library = ctypes.CDLL(str((folder/'soh_model.dll').resolve()))
    library.soh_predict.argtypes = [ctypes.POINTER(ctypes.c_float)]
    library.soh_predict.restype = ctypes.c_float
    from battery_smetimes.stm32_forecast import predict_int8
    raw = np.ascontiguousarray(raw,dtype=np.float32)
    actual = np.array([library.soh_predict(row.ctypes.data_as(ctypes.POINTER(ctypes.c_float))) for row in raw])
    expected = predict_int8(bundle['int8_layers'],bundle['scale'].transform(raw))
    error = float(np.max(np.abs(actual-expected)))
    if error > 2e-5:
        raise ValueError(f'C/Python INT8 replay mismatch {error}')
    return {'status':'passed','queries':len(raw),'max_absolute_soh_difference':error,'backend':'Windows host gcc; not STM32 board'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fold',action='append')
    parser.add_argument('--epochs',type=int,default=400)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    audit_path=Path('experiments/soh_research_v5/audits/20260930T074750_792569Z')
    frozen=json.loads((audit_path/'config.json').read_text())
    audit=json.loads((audit_path/'audit.json').read_text())
    config=frozen['config']; data_path=Path(audit['prepared_folder'])
    for path,expected in [(data_path/'manifest.json',frozen['prepared_manifest_sha256']),
                          (Path(config['integrity_audit']),frozen['integrity_sha256']),
                          (audit_path/'source_snapshot/battery_physical_registry.json',frozen['registry_sha256'])]:
        if sha(path)!=expected: raise ValueError(f'frozen artifact changed: {path}')
    manifest,cells=load_data(data_path,config['integrity_audit'])
    registry=json.loads((audit_path/'cell_registry.json').read_text())['cells']
    for key,cell in cells.items():
        cell.update({field:registry[key][field] for field in ('physical_family','reference_established_source_cycle')})
    plans=[p for p in json.loads((audit_path/'splits.json').read_text())['plans'] if p['dataset'] in ('MIT','XJTU')]
    if args.fold:
        if set(args.fold)-{p['name'] for p in plans}: raise ValueError('unknown fold')
        plans=[p for p in plans if p['name'] in args.fold]
    folder=Path('experiments/stm32_soh/runs')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    folder.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(folder/'training.log',encoding='utf-8'),logging.StreamHandler()])
    sources=['battery_smetimes/stm32_forecast.py','scripts/train_stm32_soh.py','battery_smetimes/forecast.py','battery_smetimes/research_baselines.py','scripts/train_same_type_forecast.py','docs/stm32_same_type_soh_goal.md']
    for source in sources:
        target=folder/'source_snapshot'/source; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
    save_json(folder/'config.json',{'task':'N100 same-family future SOH, compact numeric MCU predictor','seed':20261004,
        'hidden_candidates':[[32,16],[64,32]],'max_epochs':args.epochs,'raw_columns':RAW_COLUMNS,
        'selection':'validation cell-macro MAPE only; epoch picked before development refit',
        'device':args.device,'frozen_plan':str(audit_path),'source_hashes':{s:sha(s) for s in sources},
        'dataset_hashes':{r['cell_id']:r['sha256'] for r in manifest['cells']},
        'evidence':'single training seed, frozen five-fold public development CV; host C replay is not board validation'})
    save_json(folder/'splits.json',{'plans':plans})
    progress={'complete':False,'completed':[]}; save_json(folder/'progress.json',progress)
    pointer=Path('experiments/stm32_soh/latest.json'); save_json(pointer,{'folder':str(folder.resolve())})
    torch.set_num_threads(2)
    summary_rows=[]
    started=time.perf_counter()
    try:
        for plan in plans:
            logging.info('Starting %s train=%d validation=%d test=%d',plan['name'],len(plan['train']),len(plan['validation']),len(plan['test']))
            if set(plan['train'])&set(plan['validation']) or (set(plan['train'])|set(plan['validation']))&set(plan['test']): raise ValueError('overlapping cells')
            out=folder/plan['name']; out.mkdir()
            x,y,w=pairs(cells,plan['train']); scale=Scale().fit(x,w); x=scale.transform(x)
            validation=[]
            for key in plan['validation']:
                c=cells[key]; take=np.isfinite(c['soh'])&(c['soh']>0)&(c['label_cycle']>100)
                validation.append((scale.transform(compact_input(c,100,c['label_cycle'][take])),c['soh'][take]))
            candidates=[]
            for hidden in ((32,16),(64,32)):
                model,epochs,score,history=train(x,y,w,hidden,20261004,args.epochs,validation,args.device)
                name=f'candidate_{hidden[0]}_{hidden[1]}'
                torch.save(model.state_dict(),out/f'{name}.pt')
                save_json(out/f'{name}.json',{'hidden':hidden,'best_epoch':epochs,'validation_mape':score,'history':history})
                candidates.append((score,hidden,epochs))
                logging.info('%s %s validation MAPE %.3f%% epoch %d',plan['name'],hidden,score,epochs)
            score,hidden,epochs=min(candidates,key=lambda c:c[0])
            ids=plan['train']+plan['validation']; dx,dy,dw=pairs(cells,ids)
            scale=Scale().fit(dx,dw); transformed=scale.transform(dx)
            model,_,_,history=train(transformed,dy,dw,hidden,20261004,epochs,None,args.device)
            bundle={'network':model,'scale':scale,'hidden':hidden,'origins':[100],'train_ids':ids,
                'physical_family':cells[ids[0]]['physical_family'],'seed':20261004,'raw_columns':RAW_COLUMNS[plan['dataset']]}
            bundle['int8_layers']=quantize(model,transformed)
            torch.save(model.state_dict(),out/'model.pt'); joblib.dump(bundle,out/'model.joblib')
            save_json(out/'model.json',{'hidden':hidden,'epochs':epochs,'validation_mape':score,'train_ids':ids,
                'parameter_count':sum(p.numel() for p in model.parameters()),'refit_history':history})
            reports={}
            for mode in ('float','int8'):
                current=dict(bundle,quantized=mode=='int8')
                report,arrays=evaluate_baseline(current,cells,plan['test'],predictor=predict)
                save_scores(out/mode,report,arrays); reports[mode]=report
                for row in report['by_cell']:
                    summary_rows.append({'fold':plan['name'],'dataset':plan['dataset'],'mode':mode,**row})
            export_c(out/'c_export',bundle)
            checks=[]
            for key in plan['test']:
                c=cells[key]; future=c['label_cycle'][np.isfinite(c['soh'])&(c['soh']>0)&(c['label_cycle']>100)]
                raw=compact_input(c,100,future)
                checks.append(c_replay(out/'c_export',bundle,raw))
                # Prediction must be unchanged with future signals corrupted and labels absent.
                clean={k:v for k,v in c.items() if k not in ('soh','label_cycle','unfiltered_soh')}
                clean['raw']=c['raw'].copy(); clean['raw'][c['cycle']>100]=np.inf
                actual=predict(dict(bundle,quantized=True),clean,100,future)
                expected=predict(dict(bundle,quantized=True),c,100,future)
                if not np.array_equal(actual,expected): raise ValueError('future data affected prediction')
            save_json(out/'c_export/replay.json',checks)
            progress['completed'].append(plan['name']); save_json(folder/'progress.json',progress)
            save_json(folder/'by_cell.json',summary_rows)
            logging.info('%s float MAPE %.3f%% INT8 MAPE %.3f%% C replay passed',plan['name'],reports['float']['summary']['100']['cell_macro_mape_percent'],reports['int8']['summary']['100']['cell_macro_mape_percent'])
        summary={}
        for domain in ('MIT','XJTU'):
            summary[domain]={}
            for mode in ('float','int8'):
                rows=[r for r in summary_rows if r['dataset']==domain and r['mode']==mode]
                if not rows: continue
                if len({r['cell_id'] for r in rows})!=len(rows): raise ValueError('repeated outer test cell')
                mape=float(np.mean([r['metrics']['mape_percent'] for r in rows]))
                summary[domain][mode]={'cells':len(rows),'cell_macro_mape_percent':mape,
                    'fraction_cells_mape_le_6':float(np.mean([r['metrics']['mape_percent']<=6 for r in rows])),
                    'mean_late_half_mape_percent':float(np.mean([r['late_half']['mape_percent'] for r in rows])),
                    'target_passed':mape<=6}
        save_json(folder/'summary.json',summary)
        progress.update(complete=True,elapsed_seconds=time.perf_counter()-started); save_json(folder/'progress.json',progress)
        logging.info('Training completed: %s',json.dumps(summary))
    except Exception:
        save_json(folder/'failure.json',{'traceback':traceback.format_exc()}); logging.exception('Training failed'); raise


if __name__=='__main__': main()
