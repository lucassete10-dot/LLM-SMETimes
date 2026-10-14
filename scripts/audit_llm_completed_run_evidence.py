"""Read-only completion evidence audit for real completed queue runs."""
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np,torch
from threadpoolctl import threadpool_limits
from transformers import AutoConfig,AutoModel
from battery_smetimes.research_sequences import HistoryPreprocessor
from battery_smetimes.llm_reporting import raw_curve_numeric_metrics
from battery_smetimes.llm_curve_objective import curve_metrics
from scripts.train_llm_soh_mvp import evaluation_curve_metrics
from scripts.audit_llm_curve_repair import load_frozen,fingerprint
from scripts.prepare_same_type_forecast import save_json


def audit_run(task,cache,expected_roles=('training','validation','test')):
    run=Path(task['run']);cfg=json.loads((run/'config.json').read_text(encoding='utf-8'))
    result=json.loads((run/'result.json').read_text(encoding='utf-8'));verification=json.loads((run/'repair_verification.json').read_text(encoding='utf-8'))
    if result.get('post_training_recovery'):
        recovery=json.loads((run/'post_training_recovery.json').read_text(encoding='utf-8'))
        if recovery!=result['post_training_recovery'] or not recovery['checkpoint_unchanged'] or recovery['training_steps_executed']!=0:raise ValueError('post-training recovery evidence differs')
        for file,sha in recovery['source_sha256'].items():
            if fingerprint(run/'recovery_source_snapshot'/file)!=sha:raise ValueError('recovery source snapshot drift')
    archive=json.loads((run/'strict_acceptance.json').read_text(encoding='utf-8'));source=json.loads((run/'source_manifest.json').read_text(encoding='utf-8'))
    if cfg['mode']!='development' or cfg['fold']!=task['fold'] or cfg['seed']!=task['seed']:raise ValueError('run/task identity mismatch')
    actual=fingerprint(run/'model.pt')
    if any(expected!=actual for expected in (result['model_sha256'],verification['model_sha256'],task.get('model_sha256',actual))):raise ValueError('checkpoint hash mismatch')
    if not verification['passed'] or set(verification['role_scope'])!=set(expected_roles) or set(archive)!=set(expected_roles):raise ValueError('real replay/isolation audit incomplete')
    for file,expected in source['sha256'].items():
        path=Path(file);relative=path.resolve().relative_to(Path.cwd().resolve())
        if fingerprint(run/'source_snapshot'/relative)!=expected:raise ValueError('source snapshot changed: '+str(relative))
    p0=Path(cfg['p0']);p0source=p0/'source_manifest.json'
    if fingerprint(p0source)!=source['P0_manifest_sha256']:raise ValueError('P0 manifest changed')
    if str(p0) not in cache:
        pinned=json.loads(p0source.read_text(encoding='utf-8'))['checkpoint_sha256']
        for file,expected in pinned.items():
            if fingerprint(Path(file))!=expected:raise ValueError('pretrained backbone/tokenizer file changed')
        config=AutoConfig.from_pretrained(cfg['checkpoint'],local_files_only=True)
        with torch.device('meta'):structure=AutoModel.from_config(config)
        count=sum(p.numel() for p in structure.parameters());del structure
        cache[str(p0)]={'base_parameter_count':count,'pretrained_files_verified':len(pinned)}
    if result['parameters']['frozen']!=cache[str(p0)]['base_parameter_count']:raise ValueError('frozen backbone parameter count differs from pinned architecture')
    adapter=torch.load(run/'model.pt',map_location='cpu',weights_only=True)
    if any(not torch.isfinite(value).all() for value in adapter.values()):raise ValueError('nonfinite checkpoint tensor')
    if any(name.startswith('backbone.') and '.lora_' not in name for name in adapter):raise ValueError('unexpected replacement backbone weights in adapter archive')
    buffers={'query_mean','query_scale','rate_knots','local_recovery_knots'}
    if sum(value.numel() for name,value in adapter.items() if name not in buffers)!=result['parameters']['trainable']:raise ValueError('saved adapter parameter count differs from actual run declaration')
    _,cells,plan,_,_=load_frozen(cfg)
    saved=joblib.load(run/'preprocessor.joblib')
    for key,attribute in [('query_mean','mean_'),('query_scale','scale_')]:
        torch.testing.assert_close(adapter[key],torch.as_tensor(getattr(saved.query_scale.scaler,attribute),dtype=adapter[key].dtype),rtol=0,atol=0)
    if cfg['train_ids']!=plan['train'] or saved.train_ids!=plan['train'] or set(saved.raw_counts)!=set(plan['train']):raise ValueError('preprocessor TRAIN membership differs')
    fitted=HistoryPreprocessor().fit(cells,plan['train'],[cfg['origin']],history_slots=cfg['history_slots'])
    for name in ('raw_scale','stats_scale','query_scale'):
        old,new=getattr(saved,name),getattr(fitted,name)
        np.testing.assert_allclose(old.imputer.statistics_,new.imputer.statistics_,rtol=0,atol=1e-10,equal_nan=True)
        for attribute in ('mean_','var_','scale_'):np.testing.assert_allclose(getattr(old.scaler,attribute),getattr(new.scaler,attribute),rtol=0,atol=1e-10,equal_nan=True)
    diagnostics=[]
    for file in [run/'gradient_diagnostics.json',*run.glob('gradient_diagnostics_before_resume_*.json')]:
        if file.exists():diagnostics+=json.loads(file.read_text(encoding='utf-8'))
    if not any(r['gradient_norms'].get('numeric_projection.0.weight',0)>0 for r in diagnostics):raise ValueError('no positive numeric LLM-path gradient evidence')
    if cfg.get('numeric_lora_rank',0) and not any(v>0 for r in diagnostics for k,v in r['gradient_norms'].items() if k.endswith('.lora_B')):raise ValueError('LoRA gradient evidence missing')
    checked=0;roles={}
    selection=json.loads((run/'selection.json').read_text(encoding='utf-8'))
    cached_validation=json.loads((run/'selected_metrics.json').read_text(encoding='utf-8'))
    if selection['epoch']!=result['selected_epoch'] or selection['scope']!='development validation':raise ValueError('checkpoint selection scope/epoch mismatch')
    if set(cached_validation)!=set(plan['validation']):raise ValueError('cached validation membership changed')
    cached_precision_difference=0.;cached_prediction_difference=0.
    for role,record in archive.items():
        ids=plan['train' if role=='training' else role];skipped={r['cell_id'] for r in record['skipped']};evaluated=set(record['by_cell'])
        if record['cells_requested']!=len(ids) or evaluated&skipped or evaluated|skipped!=set(ids):raise ValueError('requested role coverage changed')
        probes=[r for r in verification['per_cell'] if r['role']==role]
        if {r['cell_id'] for r in probes}!=evaluated or not all(r['passed'] for r in probes):raise ValueError('model-output isolation coverage differs')
        with np.load(run/(role+'_audited_predictions.npz'),allow_pickle=False) as data:
            for cid,row in record['by_cell'].items():
                cell=cells[cid];take=(cell['label_cycle']>cfg['origin'])&np.isfinite(cell['soh'])&(cell['soh']>0)
                prefix=f'{cid}__{cfg["origin"]}__';x,y,p=[data[prefix+k] for k in ('cycle','truth','prediction')]
                np.testing.assert_array_equal(x,cell['label_cycle'][take]);np.testing.assert_allclose(y,cell['soh'][take],rtol=0,atol=1e-7)
                numeric=raw_curve_numeric_metrics(x,y,p)
                for key,value in numeric.items():
                    if key in row and isinstance(value,float):np.testing.assert_allclose(row[key],value,rtol=0,atol=1e-10)
                if role=='validation':
                    with np.load(run/'selected_predictions.npz',allow_pickle=False) as original:
                        original_arrays=[original[prefix+k] for k in ('cycle','truth','prediction')]
                    np.testing.assert_array_equal(original_arrays[0],x);np.testing.assert_allclose(original_arrays[1],y,rtol=0,atol=1e-7)
                    np.testing.assert_allclose(original_arrays[2],p,rtol=0,atol=verification['tolerance_soh'])
                    cached_prediction_difference=max(cached_prediction_difference,float(np.max(abs(original_arrays[2]-p))))
                    # The original training cache retains float32 predictions.
                    # Verify its arithmetic in that dtype; separately verify the
                    # float64 audited arithmetic without relaxing either gate.
                    if cfg.get('selection_version')=='curve_quality_v3':
                        from battery_smetimes.llm_curve_quality_v3 import curve_metrics_v3
                        original_metrics=curve_metrics_v3(*original_arrays)
                    else:original_metrics=evaluation_curve_metrics(original_arrays)
                    for key,value in numeric.items():
                        if key in cached_validation[cid] and isinstance(value,float):
                            np.testing.assert_allclose(cached_validation[cid][key],original_metrics[key],rtol=0,atol=1e-10)
                            cached_precision_difference=max(cached_precision_difference,abs(value-cached_validation[cid][key]))
                checked+=1
        roles[role]={'requested':len(ids),'audited':len(evaluated),'skipped':len(skipped)}
    return {'fold':task['fold'],'seed':task['seed'],'run':str(run.resolve()),'passed':True,'model_sha256':actual,
        'source_snapshots_verified':len(source['sha256']),'TRAIN_scalers_independently_reproduced':True,
        'numeric_gradient_evidence':True,'raw_prediction_instances_checked':checked,'roles':roles,
        'cached_validation_original_dtype_arithmetic_verified':True,'cached_vs_audited_prediction_max_error_soh':cached_prediction_difference,
        'cached_float32_vs_audited_float64_metric_max_difference_pp_or_percent':cached_precision_difference,
        'pretrained_backbone':cache[str(p0)],'not_new_accuracy_or_prediction_run':True}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--queue',type=Path,required=True);args=ap.parse_args()
    queue=json.loads((args.queue/'queue.json').read_text(encoding='utf-8'));tasks=[t for t in queue['tasks'] if t['state']=='complete']
    for state in (args.queue/'optimization').glob('*/state.json'):
        tasks.extend(t for t in json.loads(state.read_text(encoding='utf-8')).get('formal_tasks',[]) if t['state']=='complete')
    records=[];failures=[];cache={};torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        for task in tasks:
            try:record=audit_run(task,cache);records.append(record);print(json.dumps({k:record[k] for k in ('fold','seed','passed','raw_prediction_instances_checked')}),flush=True)
            except Exception as error:failures.append({'run':task['run'],'fold':task['fold'],'seed':task['seed'],'error':str(error)})
    report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'scope':'Completed real runs only; source/checkpoint provenance, TRAIN-only scaler reproduction, declared architecture count, stored gradient evidence, role coverage and raw numeric arithmetic.',
        'runs_requested':len(tasks),'runs_audited':len(records),'records':records,'failures':failures,'passed':bool(tasks and not failures and len(records)==len(tasks)),
        'source_sha256':fingerprint(Path(__file__)),'goal_complete':False,'remaining_incomplete_runs_not_certified':True}
    save_json(args.queue/'completed_run_evidence_audit.json',report)
    print(json.dumps({k:report[k] for k in ('runs_requested','runs_audited','passed','failures')},indent=2),flush=True)
    if not report['passed']:raise ValueError('completion evidence audit failed; failures retained')


if __name__=='__main__':main()
