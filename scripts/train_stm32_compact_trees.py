"""Small, family-isolated trees on frozen grouped folds with portable C replay."""
import argparse
import ctypes
import csv
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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from battery_smetimes.forecast import query_matrix
from battery_smetimes.physical_types import load_registry, require_same_registered_family
from battery_smetimes.research_baselines import build_training_pairs, signal_prefix, evaluate_baseline
from scripts.prepare_same_type_forecast import save_json, sha
from scripts.train_same_type_forecast import load_data
from scripts.train_soh_research_v5 import save_scores

AUDIT=Path('experiments/soh_research_v5/audits/20260930T074750_792569Z')
CANDIDATES=[{'trees':16,'depth':5,'leaf':4},{'trees':32,'depth':5,'leaf':4},{'trees':32,'depth':6,'leaf':8}]


def fit(cells,ids,seed,parameters,registry):
    family=cells[ids[0]]['physical_family']
    require_same_registered_family(family,[cells[k]['physical_family'] for k in ids],registry)
    x,y,w,used=build_training_pairs(cells,ids,[100])
    imputer=SimpleImputer(strategy='median',keep_empty_features=True)
    z=imputer.fit_transform(x.astype(np.float64)).astype(np.float32)
    # Supervised selection only from these training cells; keep all known
    # cycle/condition/query columns so reduced input retains operational context.
    selector=ExtraTreesRegressor(n_estimators=32,max_depth=5,min_samples_leaf=8,
                                max_features=1.,random_state=seed,n_jobs=2).fit(z,y,sample_weight=w)
    mandatory=list(range(x.shape[1]-12,x.shape[1]))
    rank=np.argsort(-selector.feature_importances_,kind='stable')
    columns=sorted(set(mandatory)|set([int(i) for i in rank if i not in mandatory][:12]))
    model=ExtraTreesRegressor(n_estimators=parameters['trees'],max_depth=parameters['depth'],
                             min_samples_leaf=parameters['leaf'],max_features=1.,random_state=seed,n_jobs=2)
    model.fit(z[:,columns],y,sample_weight=w)
    nodes=sum(t.tree_.node_count for t in model.estimators_)
    constant_bytes=nodes*12+len(model.estimators_)*2+len(columns)*2+len(columns)*4
    if constant_bytes+8192>65536:raise ValueError('estimated tree + frontend code budget exceeded')
    return dict(schema='stm32_compact_tree_v1',model=model,imputer=imputer,columns=columns,
                selector_importances=selector.feature_importances_,input_width=x.shape[1],
                physical_family=family,registry=registry,train_ids=list(ids),used_train_ids=used,
                origins=[100],seed=seed,parameters=parameters,nodes=nodes,constant_bytes=constant_bytes)


def predict(bundle,cell,origin,future):
    require_same_registered_family(cell['physical_family'],[bundle['physical_family']],bundle['registry'])
    if origin!=100:raise ValueError('only trained origin100 supported')
    x=query_matrix(signal_prefix(cell,origin),origin,future)
    z=bundle['imputer'].transform(x.astype(np.float64)).astype(np.float32)
    return bundle['model'].predict(z[:,bundle['columns']])


def literal(value):
    word=f'{float(value):.9g}'
    if '.' not in word and 'e' not in word:word+='.0'
    return word+'f'


def export(folder,bundle):
    folder.mkdir()
    nodes=[];roots=[];offset=0
    for estimator in bundle['model'].estimators_:
        tree=estimator.tree_;roots.append(offset)
        for i in range(tree.node_count):
            if tree.feature[i]<0:
                nodes.append(f'{{{literal(np.float32(tree.value[i,0,0]))},0,0,-1}}')
            else:
                threshold=np.float32(tree.threshold[i])
                # Float32 inputs against a double threshold: round down, so
                # exporting to float preserves the exact split decision.
                if float(threshold)>tree.threshold[i]:threshold=np.nextafter(threshold,np.float32(-np.inf))
                nodes.append(f'{{{literal(threshold)},{offset+tree.children_left[i]},{offset+tree.children_right[i]},{tree.feature[i]}}}')
        offset+=tree.node_count
    fills=bundle['imputer'].statistics_[bundle['columns']]
    text='#include <math.h>\n#include <stdint.h>\n#include "soh_tree.h"\n'
    text+='typedef struct { float value; uint16_t left,right; int16_t feature; } Node;\n'
    text+=f'static const Node nodes[{len(nodes)}]={{'+','.join(nodes)+'};\n'
    text+=f'static const uint16_t roots[{len(roots)}]={{'+','.join(map(str,roots))+'};\n'
    text+=f'static const uint16_t columns[{len(fills)}]={{'+','.join(map(str,bundle['columns']))+'};\n'
    text+=f'static const float fill[{len(fills)}]={{'+','.join(literal(v) for v in fills)+'};\n'
    text+=f'float soh_tree_predict(const float input[{bundle["input_width"]}]) {{\n float total=0;\n'
    text+=f'for(unsigned t=0;t<{len(roots)};t++) {{ unsigned n=roots[t]; while(nodes[n].feature>=0) {{ int f=nodes[n].feature; float x=input[columns[f]]; if(isinf(x)) return NAN; if(isnan(x)) x=fill[f]; n=x<=nodes[n].value?nodes[n].left:nodes[n].right; }} total+=nodes[n].value; }}\n'
    text+=f'return total/{literal(len(roots))}; }}\n'
    (folder/'soh_tree.c').write_text(text,encoding='utf-8')
    (folder/'soh_tree.h').write_text(f'#ifndef SOH_TREE_H\n#define SOH_TREE_H\n/* Caller must check physical family and early-input contract. */\nfloat soh_tree_predict(const float input[{bundle["input_width"]}]);\n#endif\n',encoding='utf-8')
    save_json(folder/'resources.json',dict(nodes=bundle['nodes'],trees=len(roots),selected_features=len(fills),
         input_width=bundle['input_width'],constant_bytes=bundle['constant_bytes'],reserved_code_bytes=8192,
         scope='array estimate; whole linked firmware and board latency not measured'))


def replay(folder,bundle,cells,ids):
    gcc=shutil.which('gcc')
    if not gcc:raise RuntimeError('host GCC required for C replay')
    subprocess.run([gcc,'-std=c99','-O2','-Wall','-Wextra','-Werror','-shared',str(folder/'soh_tree.c'),'-o',str(folder/'soh_tree.dll')],check=True,capture_output=True,text=True)
    lib=ctypes.CDLL(str((folder/'soh_tree.dll').resolve()))
    lib.soh_tree_predict.argtypes=[ctypes.POINTER(ctypes.c_float)];lib.soh_tree_predict.restype=ctypes.c_float
    checks=[]
    for key in ids:
        c=cells[key];take=np.isfinite(c['soh'])&(c['soh']>0)&(c['label_cycle']>100)
        future=c['label_cycle'][take]
        x=np.ascontiguousarray(query_matrix(signal_prefix(c,100),100,future),dtype=np.float32)
        actual=np.array([lib.soh_tree_predict(r.ctypes.data_as(ctypes.POINTER(ctypes.c_float))) for r in x])
        expected=predict(bundle,c,100,future);error=float(np.max(abs(actual-expected)))
        if not np.isfinite(actual).all() or error>1e-5:raise ValueError(f'C replay failed {key}: {error}')
        # Actually append poisoned future input rows, instead of assuming
        # prepared early prefixes contain any future rows already.
        clean={k:v for k,v in c.items() if k not in ('soh','label_cycle','unfiltered_soh')}
        clean['cycle']=np.r_[c['cycle'],101,200]
        clean['raw']=np.vstack([c['raw'],np.full((2,c['raw'].shape[1]),np.inf)])
        np.testing.assert_allclose(predict(bundle,clean,100,future),expected,rtol=0,atol=1e-12)
        checks.append(dict(cell_id=key,queries=len(future),max_absolute_soh_difference=error,
                           future_poison_and_label_removal=True))
    save_json(folder/'replay.json',checks)
    return checks


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    frozen=json.loads((AUDIT/'config.json').read_text());cfg=frozen['config']
    prepared=Path(json.loads((AUDIT/'audit.json').read_text())['prepared_folder'])
    registry_path=AUDIT/'source_snapshot/battery_physical_registry.json'
    for p,d in [(prepared/'manifest.json',frozen['prepared_manifest_sha256']),
                (Path(cfg['integrity_audit']),frozen['integrity_sha256']),(registry_path,frozen['registry_sha256'])]:
        if sha(p)!=d:raise ValueError('frozen inputs changed')
    registry=load_registry(registry_path);manifest,cells=load_data(prepared,cfg['integrity_audit'])
    records=json.loads((AUDIT/'cell_registry.json').read_text())['cells']
    for k,c in cells.items():c.update({f:records[k][f] for f in ('physical_family','reference_established_source_cycle')})
    plans=json.loads((AUDIT/'splits.json').read_text())['plans']
    seeds=list(range(20261004,20261009))
    if args.smoke:plans=[next(p for p in plans if p['name']=='MIT_fold1')];seeds=seeds[:1]
    out=Path('experiments/stm32_compact_trees/runs')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'training.log',encoding='utf-8'),logging.StreamHandler()])
    sources=[Path(__file__),Path('battery_smetimes/research_baselines.py'),Path('battery_smetimes/forecast.py'),Path('battery_smetimes/physical_types.py'),Path('scripts/train_same_type_forecast.py')]
    for p in sources:
        dest=out/'source_snapshot'/(Path('scripts')/p.name if p.is_absolute() else p);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    shutil.copytree(AUDIT,out/'frozen_audit')
    save_json(out/'config.json',dict(smoke=args.smoke,seeds=seeds,candidates=CANDIDATES,origin=100,
        feature_selection='train-only 32-tree selector; top12 signal stats plus12 cycle/condition/query features',
        selection='inner-validation cell-macro all-future MAPE',source_sha256={str(p):sha(p) for p in sources},
        data_sha256={r['cell_id']:r['sha256'] for r in manifest['cells']},flash_budget=65536,ram_budget=16384))
    save_json(out/'splits.json',dict(plans=plans));progress=dict(complete=False,completed=[]);save_json(out/'progress.json',progress)
    save_json(Path('experiments/stm32_compact_trees/latest.json'),dict(folder=str(out.resolve())))
    rows=[]
    try:
        for plan in plans:
            a,b,t=[set(plan[k]) for k in ('train','validation','test')]
            if a&b or (a|b)&t:raise ValueError('cell-role overlap')
            for seed in seeds:
                started=time.perf_counter();dest=out/plan['name']/str(seed);dest.mkdir(parents=True)
                scores=[]
                for i,params in enumerate(CANDIDATES):
                    candidate=dest/f'candidate_{i}';candidate.mkdir()
                    bundle=fit(cells,plan['train'],seed,params,registry);joblib.dump(bundle,candidate/'model.joblib')
                    report,arrays=evaluate_baseline(bundle,cells,plan['validation'],predictor=predict)
                    save_json(candidate/'validation_metrics.json',report);np.savez_compressed(candidate/'validation_predictions.npz',**arrays)
                    score=report['summary']['100']['cell_macro_mape_percent']
                    scores.append(dict(parameters=params,validation_mape=score))
                selected=min(scores,key=lambda r:r['validation_mape']);save_json(dest/'selection.json',dict(candidates=scores,selected=selected))
                bundle=fit(cells,plan['train']+plan['validation'],seed,selected['parameters'],registry)
                joblib.dump(bundle,dest/'model.joblib');export(dest/'c_export',bundle)
                report,arrays=evaluate_baseline(bundle,cells,plan['test'],predictor=predict);save_scores(dest/'test',report,arrays)
                checks=replay(dest/'c_export',bundle,cells,plan['test'])
                for r in report['by_cell']:
                    prefix=r['cell_id']+'__100__';x,y,p=[arrays[prefix+f] for f in ('cycle','truth','prediction')]
                    tail=x>=x[-1]-99
                    rows.append(dict(dataset=plan['dataset'],fold=plan['name'],seed=seed,cell_id=r['cell_id'],
                        **r['metrics'],late_half_mape=r['late_half']['mape_percent'],
                        last100_mape=float(100*np.mean(abs(p[tail]-y[tail])/y[tail])),endpoint_error_pp=float(abs(p[-1]-y[-1])*100),
                        nodes=bundle['nodes'],constant_bytes=bundle['constant_bytes'],metric_source=str((dest/'test/metrics.json').resolve())))
                save_json(out/'by_cell.json',rows)
                progress['completed'].append(dict(fold=plan['name'],seed=seed));save_json(out/'progress.json',progress)
                logging.info('SAVED %s seed=%s MAPE=%.3f nodes=%s constants=%s Cqueries=%s seconds=%.1f',
                    plan['name'],seed,report['summary']['100']['cell_macro_mape_percent'],bundle['nodes'],bundle['constant_bytes'],sum(c['queries'] for c in checks),time.perf_counter()-started)
        with (out/'summary.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        summary={}
        for domain in sorted({r['dataset'] for r in rows}):
            subset=[r for r in rows if r['dataset']==domain];summary[domain]={}
            for key in ('mape_percent','rmse_soh','mae_soh','r2','late_half_mape','last100_mape','endpoint_error_pp'):
                means=[float(np.mean([r[key] for r in subset if r['seed']==s])) for s in seeds]
                summary[domain][key]=dict(mean=float(np.mean(means)),sd=float(np.std(means,ddof=1)) if len(means)>1 else None)
            summary[domain].update(cells=len({r['cell_id'] for r in subset}),seeds=len(seeds),max_constant_bytes=max(r['constant_bytes'] for r in subset),
                                  fraction_cell_seed_mape_le6=float(np.mean([r['mape_percent']<=6 for r in subset])))
        save_json(out/'summary.json',summary);progress['complete']=True;save_json(out/'progress.json',progress)
        print('OUTPUT='+str(out.resolve()),flush=True)
    except BaseException:
        save_json(out/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
