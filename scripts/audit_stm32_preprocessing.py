"""Independent arithmetic checks of train-side medians, moments and INT8 calibration."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import traceback
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.external_soh import load_context
from battery_smetimes.research_baselines import build_training_pairs
from battery_smetimes.stm32_multiseed import pairs
from scripts.prepare_same_type_forecast import save_json, sha

TREE=Path('experiments/stm32_compact_trees/runs/20261004T074429_087457Z')
MLP=Path('experiments/stm32_soh_multiseed/runs/20261004T075651_692988Z')
RELEASE=Path('deployment/stm32_release_trees/20261004T083706_476897Z')


def medians(x, dtype):
    # Compute per-column directly, without using the saved scaler/imputer class.
    return np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.
                     for c in x.T], dtype=dtype)


def moments(x,w):
    fill=medians(x,np.float32)
    value=np.where(np.isfinite(x),x,fill).astype(np.float64)
    normalized=w/np.sum(w)
    mean=np.sum(value*normalized[:,None],axis=0).astype(np.float32)
    variance=np.sum((value-mean)**2*normalized[:,None],axis=0)
    std=np.where(variance>1e-12,np.sqrt(variance),1.).astype(np.float32)
    return fill,mean,std


def calibration(bundle,x):
    value=x.copy();records=[]
    for index,layer in enumerate((bundle['network'].layers[0],bundle['network'].layers[2],bundle['network'].layers[4])):
        weight=layer.weight.detach().cpu().numpy()
        bias=layer.bias.detach().cpu().numpy()
        ws=np.maximum(np.max(np.abs(weight),axis=1)/127,1e-10)
        integer=np.clip(np.rint(weight/ws[:,None]),-127,127).astype(np.int8)
        scale=max(float(np.max(np.abs(value)))/127,1e-10)
        saved=bundle['int8_layers'][index]
        np.testing.assert_array_equal(integer,saved['weight'])
        np.testing.assert_array_equal(ws.astype(np.float32),saved['weight_scale'])
        np.testing.assert_array_equal(bias,saved['bias'])
        np.testing.assert_allclose(scale,saved['input_scale'],rtol=1e-7,atol=1e-10)
        inputs=np.clip(np.rint(value/scale),-127,127).astype(np.int32)
        value=(inputs@integer.astype(np.int32).T)*(scale*ws)+bias
        if index<2:value=np.maximum(value,0)
        records.append(dict(layer=index,input_scale_recomputed=scale,input_scale_saved=saved['input_scale']))
    return records


def main():
    out=Path('experiments/stm32_preprocessing_audit')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True);shutil.copy2(__file__,out/Path(__file__).name)
    save_json(out/'config.json',dict(tree_run=str(TREE),mlp_run=str(MLP),release=str(RELEASE),
        purpose='recompute preprocessing from only declared training members, no network/tree training',
        source_sha256=sha(__file__)))
    try:
        context=load_context(RELEASE);cells=context['cells'];rows=[]
        median_cache={};pair_cache={}
        with threadpool_limits(limits=1):
            for kind,run in [('tree',TREE),('mlp',MLP)]:
                plans=json.loads((run/'splits.json').read_text())['plans']
                seeds=json.loads((run/'config.json').read_text())['seeds']
                for plan in plans:
                    for seed in seeds:
                        dest=run/plan['name']/str(seed)
                        paths=[dest/'model.joblib']
                        if kind=='tree':paths+=sorted(dest.glob('candidate_*/model.joblib'))
                        for path in paths:
                            bundle=joblib.load(path);ids=bundle['train_ids'];key=tuple(ids)
                            allowed=plan['train'] if 'candidate_' in str(path) else plan['train']+plan['validation']
                            if set(ids)!=set(allowed) or set(ids)&set(plan['test']):
                                raise ValueError('training member leakage')
                            row=dict(kind=kind,fold=plan['name'],seed=seed,model=str(path),model_sha256=sha(path),
                                training_cells=len(ids),test_cells_used=0)
                            if kind=='tree':
                                if key not in median_cache:
                                    x,_,_,used=build_training_pairs(cells,ids,[100])
                                    if set(used)!=set(ids):raise ValueError('training members silently skipped')
                                    median_cache[key]=medians(x.astype(np.float64),np.float64)
                                actual=median_cache[key]
                                np.testing.assert_allclose(bundle['imputer'].statistics_,actual,rtol=0,atol=0)
                                row.update(statistics='all-column medians match exactly',columns=len(actual))
                            else:
                                if key not in pair_cache:pair_cache[key]=pairs(cells,ids)
                                x,_,w=pair_cache[key]
                                fill,mean,std=moments(x,w)
                                scale=bundle['scale']
                                np.testing.assert_array_equal(fill,scale.fill)
                                np.testing.assert_allclose(mean,scale.mean,rtol=1e-7,atol=1e-7)
                                np.testing.assert_allclose(std,scale.std,rtol=1e-7,atol=1e-7)
                                # Saved transform is not called: apply independently recomputed parameters.
                                transformed=np.clip((np.where(np.isfinite(x),x,fill)-mean)/std,-8,8).astype(np.float32)
                                row.update(statistics='fill/weighted mean/std match',columns=len(fill),
                                    quantization=calibration(bundle,transformed))
                            rows.append(row)
                    save_json(out/'progress.json',dict(complete=False,checked=len(rows),last_fold=plan['name']))
                    print(kind,plan['name'],'PASS',len(rows),flush=True)
        save_json(out/'checks.json',rows)
        save_json(out/'progress.json',dict(complete=True,passed=True,checked=len(rows),
            tree_models=sum(r['kind']=='tree' for r in rows),mlp_models=sum(r['kind']=='mlp' for r in rows),
            scope='Saved final-fold and tree-inner imputation, final-fold MLP normalization and quantization. Historical MLP inner scalers not serialized; no reconstruction claim for those. No board or independent-cell accuracy claim.'))
        print('OUTPUT='+str(out.resolve()))
    except BaseException:
        save_json(out/'failure.json',dict(traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
