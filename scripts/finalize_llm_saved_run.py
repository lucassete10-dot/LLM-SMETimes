"""Recover only post-training diagnostics; never take an optimizer step."""
import argparse,json,time,shutil
from pathlib import Path
import joblib,numpy as np,torch
from scripts.train_llm_curve_repair import create_network,summary,tiny_gate
from scripts.train_llm_soh_mvp import evaluate
from scripts.audit_llm_curve_repair import load_frozen,fingerprint
from scripts.prepare_same_type_forecast import save_json
from battery_smetimes.research_llm import load_adapter
from battery_smetimes.physical_types import load_registry

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args();root=args.run
    cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
    selection=json.loads((root/'selection.json').read_text(encoding='utf-8'))
    saved=torch.load(root/'resume.pt',map_location='cpu',weights_only=False)
    if saved['epoch']!=cfg['max_tiny_epochs']:raise ValueError('Only fully trained runs can use this finalizer')
    model_sha=fingerprint(root/'model.pt');start=time.perf_counter()
    _,cells,_,registry_path,_=load_frozen(cfg);processor=joblib.load(root/'preprocessor.joblib')
    if processor.train_ids!=cfg['train_ids']:raise ValueError('TRAIN scaler identity mismatch')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    net=create_network(cfg,processor,cfg['architecture'],cfg['dropout'])
    load_adapter(net,torch.load(root/'model.pt',map_location='cpu',weights_only=True));net.eval()
    bundle={'network':net,'processor':processor,'device':'cuda','origins':[cfg['origin']],
        'registry':load_registry(registry_path),'physical_family':cells[cfg['train_ids'][0]]['physical_family']}
    _,arrays,metrics,_=evaluate(bundle,cells,cfg['train_ids']);gates=tiny_gate(metrics,cfg)
    save_json(root/'TRAIN_metrics.json',metrics);np.savez_compressed(root/'TRAIN_predictions.npz',**arrays)
    assert fingerprint(root/'model.pt')==model_sha
    recovery_sources=[Path(__file__),Path('scripts/train_llm_soh_mvp.py'),Path('scripts/train_llm_curve_repair.py')]
    for source in recovery_sources:
        target=root/'recovery_source_snapshot'/source.resolve().relative_to(Path.cwd().resolve())
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    evidence={'checkpoint_unchanged':True,'training_steps_executed':0,'completed_training_epoch':saved['epoch'],
        'source_sha256':{str(p.resolve().relative_to(Path.cwd().resolve())):fingerprint(p) for p in recovery_sources},
        'finalizer_sha256':fingerprint(Path(__file__)),'evaluation_helper_sha256':fingerprint(Path('scripts/train_llm_soh_mvp.py')),
        'summary_helper_sha256':fingerprint(Path('scripts/train_llm_curve_repair.py')),'reason':'Recover missing TRAIN stage metrics after completed training; retain original source snapshots and failure record'}
    save_json(root/'post_training_recovery.json',evidence)
    save_json(root/'result.json',{'selected_epoch':selection['epoch'],'parameters':net.parameter_counts(),
        'TRAIN_summary':summary(metrics),'tiny_per_cell_pass':gates,'tiny_all_pass':all(gates.values()),
        'cuda_peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20,'elapsed_seconds':None,
        'post_training_recovery_seconds':time.perf_counter()-start,'model_sha256':model_sha,
        'not_generalization_accuracy':False,'post_training_recovery':evidence,'goal_complete':False})
    save_json(root/'status.json',{'state':'complete','training_complete':True,'mode':cfg['mode'],
        'best_epoch':selection['epoch'],'post_training_recovered':True,'goal_complete':False})
    print(json.dumps(evidence,indent=2),flush=True)

if __name__=='__main__':main()
