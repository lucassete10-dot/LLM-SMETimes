"""Export approved full-signal estimators and public, machine-independent results."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import zipfile
import joblib
import numpy as np
from battery_smetimes.llm_full_life import read
from scripts.audit_llm_curve_repair import fingerprint,load_frozen
from scripts.infer_full_signal_soh import load_processor
from scripts.train_llm_full_life import save


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train',type=Path,default=Path('experiments/llm_reconstruction/20261016_TRAIN'))
    ap.add_argument('--mvp',type=Path,default=Path('experiments/llm_reconstruction/20261016_heldout_MVP'))
    ap.add_argument('--project',type=Path,default=Path('projects/full_signal_soh'))
    ap.add_argument('--out',type=Path,default=Path('deliveries/full_signal_soh_v0.1'))
    args=ap.parse_args();project=args.project;out=args.out
    project.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    audit=read(args.train/'selected_delivery_audit.json');mvp=read(args.mvp/'result.json')
    assert audit['passed'] and read(args.mvp/'delivery_audit.json')['passed']
    summaries=[];published=[]
    for chosen in audit['chosen']:
        ds=chosen['dataset'];run=Path(chosen['run']);cfg=read(run/'config.json')
        index=read(cfg['index']);_,cells,plan,_,_=load_frozen(index['base_settings'])
        cid=plan['validation'][0];cell=cells[cid]
        baseline=Path(cfg.get('parent_run',str(run)));processor=joblib.load(baseline/'processor.joblib')
        dest=out/'estimators'/ds;dest.mkdir(parents=True,exist_ok=True)
        assert fingerprint(run/'model.pt')==chosen['model_sha256']
        shutil.copy2(run/'model.pt',dest/'model.pt')
        metadata={k:v for k,v in cell['metadata'].items() if k in ('dataset','signal_schema')}
        backbone=Path(cfg['checkpoint'])
        filenames=['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json']
        filenames += [p.name for p in sorted(backbone.glob('*.safetensors'))]
        hashes={n:fingerprint(backbone/n) for n in filenames if (backbone/n).exists()}
        config=dict(dataset=ds,seed=cfg['seed'],raw_width=processor.raw_width,metadata=metadata,
            scaling=dict(median=processor.scale.imputer.statistics_.tolist(),
                         mean=processor.scale.scaler.mean_.tolist(),scale=processor.scale.scaler.scale_.tolist()),
            model_sha256=chosen['model_sha256'],backbone_files_sha256=hashes,
            backbone_id='meta-llama/Llama-3.2-1B-Instruct',task='full-signal same-time SOH estimation',
            causal_online=False,future_forecasting_supported=False,
            feature_extraction_source='scripts/prepare_llm_full_life.py',train_cells=len(cfg['train_ids']))
        portable=load_processor(config)
        extraction=read(Path('data/interim/llm_reconstruction_mvp/20261016_heldout')/ds/'index.json')['cells'][0]
        with np.load(extraction['path'],allow_pickle=False) as raw:
            np.testing.assert_array_equal(portable.scale.transform(raw['raw']),processor.scale.transform(raw['raw']))
        save(dest/'estimator.json',config)
        pub=project/'examples'/ds;pub.mkdir(parents=True,exist_ok=True)
        for mode in ('full_signal','prefix_only'):
            shutil.copy2(args.mvp/ds/(mode+'.png'),pub/(mode+'.png'))
        row=next(r for r in mvp['cells'] if r['dataset']==ds)
        with np.load(args.mvp/ds/'predictions.npz',allow_pickle=False) as data:
            with (pub/'curves.csv').open('w',newline='',encoding='utf-8') as f:
                w=csv.writer(f);w.writerow(['source_cycle','observed_soh','full_signal_soh','untrained_prefix_extrapolation'])
                w.writerows(zip(data['cycle'],data['truth'],data['full_signal_prediction'],data['prefix_only_extrapolation']))
        summaries.append(dict(dataset=ds,train_cells=len(cfg['train_ids']),
            train_mape_percent=chosen['mean_mape_percent'],train_mae_pp=chosen['mean_mae_pp'],
            validation_cell=cid,validation_full_signal=row['full_signal_metrics'],
            validation_untrained_prefix_extrapolation=row['prefix_only_metrics'],
            prefix_observations=row['prefix_observations'],model_sha256=chosen['model_sha256']))
        shutil.copy2(Path(chosen['report'])/'all_cells.pdf',out/(ds+'_TRAIN_curves.pdf'))
        for name in ('model.pt','estimator.json'):
            p=dest/name;published.append(dict(path=p.relative_to(out).as_posix(),sha256=fingerprint(p),bytes=p.stat().st_size))
    shutil.copy2(args.train/'selected_TRAIN_metrics.csv',project/'TRAIN_cell_metrics.csv')
    shutil.copy2(args.mvp/'all_MVP_curves.pdf',out/'all_MVP_curves.pdf')
    save(project/'results.json',dict(task='full-signal SOH estimation',seed=20260930,train_cells=240,
        validation_cells=5,no_claim_of_forecasting=True,results=summaries))
    save(out/'manifest.json',dict(format_version=1,files=published,backbone_included=False,
        dataset_raw_files_included=False,scaling_json_replay_exact=True))
    with zipfile.ZipFile(out/'full_signal_soh_estimators_v0.1.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        z.write(out/'manifest.json','manifest.json')
        for p in sorted((out/'estimators').rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out))
    print(dict(estimators=len(summaries),asset=str(out/'full_signal_soh_estimators_v0.1.zip')),flush=True)


if __name__=='__main__':main()
