"""Refit deployable trees on all public development cells; no test metric claim.

Hyperparameters use only saved inner-validation candidate scores, not outer
test scores. Original five-seed/fold results remain immutable. Independent
new-cell evaluation is required for these final weights.
"""
import argparse
from datetime import datetime,timezone
import json
import logging
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import joblib
import numpy as np
from battery_smetimes.forecast import query_matrix
from battery_smetimes.research_baselines import signal_prefix
from battery_smetimes.physical_types import load_registry
from scripts.prepare_same_type_forecast import save_json,sha
from scripts.train_same_type_forecast import load_data
from scripts.train_stm32_compact_trees import fit,export,replay
from scripts.package_stm32_compact_trees import frontend


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args();run=args.run
    if not json.loads((run/'progress.json').read_text())['complete']:raise ValueError('CV run incomplete')
    frozen=run/'frozen_audit';contract=json.loads((frozen/'config.json').read_text());prepared=Path(json.loads((frozen/'audit.json').read_text())['prepared_folder'])
    registry_file=frozen/'source_snapshot/battery_physical_registry.json'
    for file,digest in [(prepared/'manifest.json',contract['prepared_manifest_sha256']),(registry_file,contract['registry_sha256']),(Path(contract['config']['integrity_audit']),contract['integrity_sha256'])]:
        if sha(file)!=digest:raise ValueError('frozen inputs changed')
    _,cells=load_data(prepared,contract['config']['integrity_audit']);records=json.loads((frozen/'cell_registry.json').read_text())['cells']
    for key,c in cells.items():c.update({f:records[key][f] for f in ('physical_family','reference_established_source_cycle')})
    registry=load_registry(registry_file);cfg=json.loads((run/'config.json').read_text());seed=cfg['seeds'][0]
    plans=json.loads((run/'splits.json').read_text())['plans']
    out=Path('deployment/stm32_release_trees')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True)
    shutil.copytree(frozen,out/'frozen_audit')
    for file in [Path(__file__),Path('scripts/train_stm32_compact_trees.py'),Path('scripts/package_stm32_compact_trees.py')]:shutil.copy2(file,out/file.name)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'training.log',encoding='utf-8'),logging.StreamHandler()])
    save_json(out/'config.json',dict(cv_run=str(run.resolve()),seed=seed,origin=100,selection='mean saved inner-validation candidate MAPE, no outer-test scores; predetermined first seed for release',
        evaluation='all public cells become development data; do not report training fit as held-out accuracy',source_sha256={str(file):sha(file) for file in [Path(__file__),Path('scripts/train_stm32_compact_trees.py'),Path('scripts/package_stm32_compact_trees.py')]}))
    reports=[]
    try:
        for domain in ('MIT','XJTU','NASA'):
            started=time.perf_counter();scores={};sources=[]
            for plan in [p for p in plans if p['dataset']==domain]:
                for s in cfg['seeds']:
                    file=run/plan['name']/str(s)/'selection.json';selection=json.loads(file.read_text());sources.append(dict(file=str(file.resolve()),sha256=sha(file)))
                    for candidate in selection['candidates']:
                        key=json.dumps(candidate['parameters'],sort_keys=True);scores.setdefault(key,[]).append(candidate['validation_mape'])
            selected=min(scores,key=lambda k:float(np.mean(scores[k])));parameters=json.loads(selected)
            ids=sorted(k for k,c in cells.items() if c['metadata']['dataset']==domain)
            bundle=fit(cells,ids,seed,parameters,registry);dest=out/domain;export(dest,bundle);raw_cols=frontend(dest,bundle)
            joblib.dump(bundle,dest/'model.joblib')
            selected_info=dict(parameters=parameters,inner_candidate_means={k:float(np.mean(v)) for k,v in scores.items()},sources=sources)
            save_json(dest/'selection.json',selected_info)
            save_json(dest/'model_provenance.json',dict(train_ids=ids,physical_family=bundle['physical_family'],nodes=bundle['nodes'],constant_bytes=bundle['constant_bytes'],seed=seed,
                evaluation='independent new-cell accuracy not yet measured; historical CV numbers refer to fold models',train_data_sha256={k:records[k]['prepared_sha256'] for k in ids}))
            subprocess.run([shutil.which('gcc'),'-std=c99','-O2','-Wall','-Wextra','-Werror','-shared',str(dest/'soh_tree.c'),str(dest/'soh_stream.c'),'-o',str(dest/'pipeline.dll'),'-lm'],check=True,capture_output=True,text=True)
            checks=replay(dest,bundle,cells,ids);max_target=100
            for key in ids:
                cell=cells[key];future=cell['label_cycle'][np.isfinite(cell['soh'])&(cell['soh']>0)&(cell['label_cycle']>100)]
                max_target=max(max_target,int(future[-1]))
            save_json(dest/'release_numeric_replay.json',dict(checks=checks,scope='C inference numeric replay on development inputs, not generalization metrics or raw app replay'))
            save_json(dest/'supported_horizon.json',dict(origin=100,largest_development_target_source_cycle=max_target,
                rule='Warn for requests beyond development support; this is not the new cell true lifetime/EOL, and cannot be used to classify its lifetime'))
            report=dict(dataset=domain,cells=len(ids),parameters=parameters,nodes=bundle['nodes'],constant_bytes=bundle['constant_bytes'],training_seconds=time.perf_counter()-started)
            reports.append(report);save_json(out/'progress.json',dict(complete=False,reports=reports));logging.info('SAVED %s release %s',domain,report)
        save_json(out/'manifest.json',dict(reports=reports,source_cv_run=str(run.resolve()),files_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},
            pending=['final-weight raw app replay','reference linkage resource check','actual STM32 board validation','independent new-cell evaluation']))
        save_json(out/'progress.json',dict(complete=True,reports=reports));print('OUTPUT='+str(out.resolve()))
    except BaseException:
        save_json(out/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
