"""Export interchange models, label-free examples, and a verified handoff manifest."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import joblib
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from onnx.reference import ReferenceEvaluator
from battery_smetimes.stm32_forecast import compact_input, predict
from scripts.prepare_same_type_forecast import save_json, sha
from scripts.train_same_type_forecast import load_data


def export_onnx(bundle, path, quantized):
    nodes=[]; values=[]
    def tensor(name,value):
        values.append(numpy_helper.from_array(np.asarray(value),name)); return name
    for name in ('fill','mean','std'):
        tensor(name,getattr(bundle['scale'],name).astype(np.float32))
    tensor('clip_lo',np.array(-8,dtype=np.float32)); tensor('clip_hi',np.array(8,dtype=np.float32))
    nodes.extend([helper.make_node('IsNaN',['features'],['missing']),
                  helper.make_node('Where',['missing','fill','features'],['filled']),
                  helper.make_node('Sub',['filled','mean'],['centered']),
                  helper.make_node('Div',['centered','std'],['normalized']),
                  helper.make_node('Clip',['normalized','clip_lo','clip_hi'],['x0'])])
    for index,layer in enumerate([bundle['network'].layers[i] for i in (0,2,4)]):
        x=f'x{index}'; w=f'w{index}'; bias=f'b{index}'; tensor(bias,layer.bias.detach().numpy())
        if quantized:
            params=bundle['int8_layers'][index]
            tensor(w,params['weight']); tensor(f'ws{index}',params['weight_scale'])
            tensor(f'wz{index}',np.zeros(len(params['weight_scale']),dtype=np.int8))
            tensor(f'xs{index}',np.array(params['input_scale'],dtype=np.float32))
            tensor(f'xz{index}',np.array(0,dtype=np.int8))
            tensor(f'qlo{index}',np.array(-127*params['input_scale'],dtype=np.float32))
            tensor(f'qhi{index}',np.array(127*params['input_scale'],dtype=np.float32))
            nodes.extend([helper.make_node('Clip',[x,f'qlo{index}',f'qhi{index}'],[f'bounded{index}']),
                helper.make_node('QuantizeLinear',[f'bounded{index}',f'xs{index}',f'xz{index}'],[f'qx{index}']),
                helper.make_node('DequantizeLinear',[f'qx{index}',f'xs{index}',f'xz{index}'],[f'dx{index}']),
                helper.make_node('DequantizeLinear',[w,f'ws{index}',f'wz{index}'],[f'dw{index}'],axis=0)])
            x,w=f'dx{index}',f'dw{index}'
        else:
            tensor(w,layer.weight.detach().numpy())
        nodes.append(helper.make_node('Gemm',[x,w,bias],[f'z{index}'],transB=1))
        if index<2: nodes.append(helper.make_node('Relu',[f'z{index}'],[f'x{index+1}']))
    tensor('offset',np.array(1,dtype=np.float32)); nodes.append(helper.make_node('Add',['z2','offset'],['soh']))
    graph=helper.make_graph(nodes,'compact_soh_int8' if quantized else 'compact_soh_float',
        [helper.make_tensor_value_info('features',TensorProto.FLOAT,[None,40])],
        [helper.make_tensor_value_info('soh',TensorProto.FLOAT,[None,1])],values)
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',19)],ir_version=9)
    onnx.checker.check_model(model); onnx.save(model,path)


def main():
    run=Path(json.loads(Path('experiments/stm32_soh/latest.json').read_text())['folder'])
    output=Path('deployment/stm32_soh')/run.name
    audit=Path('experiments/soh_research_v5/audits/20260930T074750_792569Z')
    config=json.loads((audit/'config.json').read_text())['config']
    _,cells=load_data(Path(json.loads((audit/'audit.json').read_text())['prepared_folder']),config['integrity_audit'])
    registered=json.loads((audit/'cell_registry.json').read_text())['cells']
    for key,c in cells.items():
        c.update({f:registered[key][f] for f in ('physical_family','reference_established_source_cycle')})
    plans=json.loads((run/'splits.json').read_text())['plans']; reports=[]
    for domain in ('MIT','XJTU'):
        dest=output/domain; plan=next(p for p in plans if p['name']==f'{domain}_fold1')
        bundle=joblib.load(dest/'model.joblib'); key=plan['test'][0]; cell=cells[key]
        example=dest/'example'; example.mkdir(exist_ok=True)
        np.savez_compressed(example/'observed_signals.npz',cycle=cell['cycle'][cell['cycle']<=100],raw_features=cell['raw'][cell['cycle']<=100])
        save_json(example/'metadata.json',{'cell_id':key,'metadata':cell['metadata'],
            'physical_family':cell['physical_family'],'reference_established_source_cycle':cell['reference_established_source_cycle']})
        cmd=[sys.executable,'-m','scripts.predict_stm32_soh','--model',str(dest/'model.joblib'),
            '--signals',str(example/'observed_signals.npz'),'--metadata',str(example/'metadata.json'),
            '--end-cycle','200','--output',str(example/'predicted_soh.json')]
        subprocess.run(cmd,check=True)
        result=json.loads((example/'predicted_soh.json').read_text()); future=np.array(result['future_source_cycles'])
        expected=predict(dict(bundle,quantized=True),cell,100,future)
        if not np.array_equal(expected,np.array(result['predicted_soh'],dtype=np.float32)): raise ValueError('CLI output differs')
        # This example contains signal observations and metadata only.
        with np.load(example/'observed_signals.npz') as signals:
            if set(signals.files)!={'cycle','raw_features'}: raise ValueError('labels in example')
        negative=[]
        for field,value in [('physical_family','unsupported_family')]:
            bad=dict(cell); bad[field]=value
            try: predict(bundle,bad,100,future)
            except ValueError: negative.append(field)
            else: raise ValueError('incompatible battery accepted')
        for mode in ('float','int8'):
            model=dest/f'model_{mode}.onnx'; export_onnx(bundle,model,mode=='int8')
            evaluator=ReferenceEvaluator(str(model)); error=0; count=0
            for test_key in plan['test']:
                test_cell=cells[test_key]
                cycles=test_cell['label_cycle'][np.isfinite(test_cell['soh'])&(test_cell['soh']>0)&(test_cell['label_cycle']>100)]
                queries=compact_input(test_cell,100,cycles)
                actual=evaluator.run(None,{'features':queries})[0][:,0]
                expected=predict(dict(bundle,quantized=mode=='int8'),test_cell,100,cycles)
                error=max(error,float(np.max(np.abs(actual-expected)))); count+=len(cycles)
                if error>2e-5: raise ValueError(f'ONNX discrepancy {mode}/{test_key}: {error}')
            reports.append({'dataset':domain,'mode':mode,'file':str(model),'max_absolute_soh_difference':error,
                'cells':len(plan['test']),'queries':count,
                'verification':'ONNX checker and reference evaluator; all fold1 holdout prefixes and complete observed-future queries'})
        (example/'command.ps1').write_text('.\\.venv\\Scripts\\python.exe -m scripts.predict_stm32_soh --model "'+str(dest/'model.joblib')+'" --signals "'+str(example/'observed_signals.npz')+'" --metadata "'+str(example/'metadata.json')+'" --end-cycle 200 --output "'+str(example/'predicted_soh.json')+'"\n')
    shutil.copy2('scripts/predict_stm32_soh.py',output/'predict_stm32_soh_source.py')
    shutil.copy2(__file__,output/'package_source.py')
    save_json(output/'onnx_and_cli_verification.json',{'status':'passed','onnx_version':onnx.__version__,
        'reports':reports,'incompatible_type_rejected':True,
        'interchange_INT8':'per-channel INT8 weights and per-tensor activation QDQ; validate target ST tool version before board generation'})
    (output/'deployment_requirements.txt').write_text('onnx=='+onnx.__version__+'\n')
    files={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and p.name!='handoff_manifest.json'}
    save_json(output/'handoff_manifest.json',{'training_run':str(run.resolve()),'files_sha256':files,
        'hardware_status':'No DFU, ST-Link, J-Link or serial port connected at software handoff',
        'board_remaining':['choose actual STM32 model','verify target tool operator support or use portable C','board link/startup/ADC integration','measure board Flash/RAM/latency','board signal replay and real-cell capacity validation']})
    print('Verified label-free CLI and four float/INT8 ONNX models; handoff manifest saved')


if __name__=='__main__': main()
