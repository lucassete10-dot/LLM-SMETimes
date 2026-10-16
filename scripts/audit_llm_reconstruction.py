"""Independent TRAIN reconstruction delivery checks; no accuracy laundering."""
import argparse
import csv
from pathlib import Path
import numpy as np
import pymupdf
import torch
from battery_smetimes.llm_full_life import read,load_full_life
from battery_smetimes.llm_reconstruction import ReconstructionProcessor,add_targets
from battery_smetimes.llm_curve_quality_v3 import curve_metrics_v3
from scripts.train_llm_full_life import save
from scripts.audit_llm_curve_repair import fingerprint


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);root=ap.parse_args().root
    queue=read(root/'queue.json')
    if not queue['complete'] or len(queue['tasks'])!=5:raise ValueError('Five TRAIN datasets incomplete')
    for path,sha in read(root/'registration.json').items():
        if fingerprint(path)!=sha:raise ValueError('Registered source changed')
    total=0;datasets=[];expected={}
    for task in queue['tasks']:
        ds=task['dataset'];run=root/'runs'/ds;report=root/'reports'/ds
        cfg=read(run/'config.json');result=read(run/'result.json');replay=read(run/'replay_audit.json')
        if task['state']!='complete' or not replay['passed'] or fingerprint(run/'model.pt')!=replay['model_sha256']:
            raise ValueError('Model replay missing')
        if fingerprint(cfg['index'])!=cfg['index_sha256']:raise ValueError('Full-life index changed')
        _,cells,plan,_=load_full_life(cfg['index'])
        if cfg['train_ids']!=plan['train'] or cfg['validation_test_used']:raise ValueError('Split drift')
        sources=read(run/'source_manifest.json')
        for path,sha in sources.items():
            if fingerprint(path)!=sha:raise ValueError('Model source changed')
            relative=Path(path).relative_to(Path.cwd())
            if fingerprint(run/'source_snapshot'/relative)!=sha:raise ValueError('Source snapshot differs')
        if result['joint_parameters']['frozen']<1_000_000_000:raise ValueError('Pretrained backbone evidence absent')
        cache=read(run/'encoded_manifest.json')
        if fingerprint(run/'encoded_tokens.pt')!=cache['sha256'] or fingerprint(run/'joint_model.pt')!=cache['encoder_adapter_sha256']:
            raise ValueError('Learned feature cache provenance differs')
        joint=torch.load(run/'joint_model.pt',map_location='cpu',weights_only=True)
        selected=torch.load(run/'model.pt',map_location='cpu',weights_only=True)
        for key in joint:
            if not key.startswith(('reconstruction_','slot_embedding.')):
                torch.testing.assert_close(joint[key],selected[key],rtol=0,atol=0)
        gradients=read(run/'gradients.json')
        if len(gradients)!=12:raise ValueError('Joint learning budget differs')
        for epoch in gradients:
            for prefix in ('numeric_projection.','experts.','reconstruction_head.'):
                if not any(k.startswith(prefix) and np.isfinite(v) and v>0 for k,v in epoch['norms'].items()):
                    raise ValueError('Missing live learning gradient')
        manifest=read(report/'figure_manifest.json');visual=read(report/'visual_review.json')
        if any(q['exit_code'] for q in manifest['qa']) or manifest['visual_review']['state']!='completed' or visual['pending_figures']:
            raise ValueError('Figure review incomplete')
        if {r['cell_id'] for r in visual['figures']}!=set(plan['train']):raise ValueError('Reviewed cell coverage differs')
        for row in visual['figures']:
            if fingerprint(row['png'])!=row['sha256']:raise ValueError('Inspected PNG changed')
        if fingerprint(report/'visual_review.json')!=manifest['visual_review']['review_sha256']:raise ValueError('Review evidence changed')
        if fingerprint(report/'all_cells.pdf')!=manifest['pdf_sha256']:raise ValueError('PDF changed')
        with pymupdf.open(report/'all_cells.pdf') as pdf:
            if pdf.page_count!=len(plan['train']):raise ValueError('PDF page coverage differs')
        metrics=read(run/'metrics.json');processor=ReconstructionProcessor().fit(cells,plan['train'])
        signal_coverage=[]
        with np.load(run/'predictions.npz',allow_pickle=False) as arrays:
            if len(arrays)!=3*len(plan['train']):raise ValueError('Prediction array coverage differs')
            for cid in plan['train']:
                x,y,p=(arrays[cid+'__'+k] for k in ('cycle','truth','prediction'))
                c=cells[cid];valid=np.isfinite(c['soh'])&(c['soh']>0)
                np.testing.assert_array_equal(x,c['label_cycle'][valid]);np.testing.assert_array_equal(y,c['soh'][valid])
                actual=curve_metrics_v3(x,y,p)
                for k,v in actual.items():
                    if isinstance(v,(float,int)) and not isinstance(v,bool):np.testing.assert_allclose(v,metrics[cid][k],atol=1e-9,rtol=0)
                expected[(ds,cid)]=(actual,replay['model_sha256'])
                inputs=processor.inputs(c);stripped={k:v for k,v in c.items() if k not in ('soh','label_cycle','capacity','eol','lifetime')}
                for a,b in zip(inputs,processor.inputs(stripped)):
                    np.testing.assert_array_equal(a['history'],b['history']);assert a['card']==b['card']
                contexts=add_targets(inputs,c);used=set()
                for r in contexts:used.update(r['signal_cycles'].tolist())
                signal_coverage.append(dict(cell_id=cid,full_signal_observations=len(c['cycle']),
                    actual_signal_observations_in_supervised_contexts=len(used),all_valid_labels=len(y)))
        score=float(np.mean([m['mae_pp'] for m in metrics.values()]))
        np.testing.assert_allclose(score,result['TRAIN_mae_pp'],atol=1e-9,rtol=0)
        total+=len(plan['train']);datasets.append(dict(dataset=ds,cells=len(plan['train']),model_sha256=replay['model_sha256'],
            mean_mae_pp=score,mean_mape_percent=result['TRAIN_mape_percent'],signal_coverage=signal_coverage,
            pdf_sha256=manifest['pdf_sha256'],visual_review_sha256=fingerprint(report/'visual_review.json')))
    if total!=240:raise ValueError('All 240 TRAIN cells required')
    with (root/'all_TRAIN_metrics.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if len(rows)!=240 or {(r['dataset'],r['cell_id']) for r in rows}!=set(expected):raise ValueError('CSV coverage differs')
    for r in rows:
        m,sha=expected[(r['dataset'],r['cell_id'])]
        if r['model_sha256']!=sha:raise ValueError('CSV model differs')
        for k in ('mape_percent','mae_pp','tail_mae_pp','endpoint_error_pp'):
            np.testing.assert_allclose(float(r[k]),m[k],atol=1e-9,rtol=0)
    save(root/'delivery_audit.json',dict(passed=True,TRAIN_cells=total,datasets=datasets,
        actual_all_figures_reviewed=True,raw_labels_exact=True,TRAIN_only=True,
        reconstruction_not_forecasting=True,accuracy_success_not_implied=True,auditor_sha256=fingerprint(__file__)))
    print(dict(passed=True,TRAIN_cells=total,datasets=len(datasets)))


if __name__=='__main__':main()
