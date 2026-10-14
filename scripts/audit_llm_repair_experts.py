"""Audit trained router saturation on frozen TRAIN prefixes without target labels."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json
from pathlib import Path
import joblib,numpy as np,torch
from battery_smetimes.research_llm import load_adapter,cards_embedding
from battery_smetimes.physical_types import load_registry
from scripts.audit_llm_curve_repair import load_frozen,fingerprint
from scripts.train_llm_curve_repair import create_network
from scripts.prepare_same_type_forecast import save_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args();root=args.run
    cfg=json.loads((root/'config.json').read_text());_,cells,_,registry,_=load_frozen(cfg)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    processor=joblib.load(root/'preprocessor.joblib');net=create_network(cfg,processor,cfg['architecture'],cfg['dropout'])
    load_adapter(net,torch.load(root/'model.pt',map_location='cpu',weights_only=True));net.eval()
    bundle={'network':net,'processor':processor,'device':'cuda','registry':load_registry(registry),
            'physical_family':cells[cfg['train_ids'][0]]['physical_family']}
    captured={};handle=net.gate.register_forward_hook(lambda module,inputs,output:captured.update(logits=output.detach()))
    rows=[]
    try:
        with torch.inference_mode():
            for cid in cfg['train_ids']:
                value=processor.model_input(cells[cid],cfg['origin']);embedding=cards_embedding(bundle,cells[cid],cfg['origin'])[None].cuda()
                encoded=net.encode(torch.as_tensor(value['history'][None],device='cuda'),torch.tensor([value['length']],device='cuda'),embedding)
                valid=encoded[1] if isinstance(encoded,tuple) else torch.ones(captured['logits'].shape[:2],device='cuda',dtype=torch.bool)
                logits=captured['logits'][valid];gate=logits.softmax(-1);entropy=-(gate*gate.clamp_min(1e-30).log()).sum(-1)
                rows.append({'cell':cid,'expert_usage':gate.mean(0).cpu().tolist(),
                    'mean_router_entropy_nats':float(entropy.mean()),'maximum_probability_mean':float(gate.max(-1).values.mean()),
                    'fraction_tokens_above_0999_probability':float((gate.max(-1).values>.999).float().mean()),
                    'max_absolute_logit':float(logits.abs().max())})
    finally:handle.remove()
    report={'scope':'TRAIN visible inputs only; no target curve or model selection',
        'model_sha256':fingerprint(root/'model.pt'),'source_sha256':fingerprint(Path(__file__)),
        'by_cell':rows,'mean_expert_usage':np.mean([r['expert_usage'] for r in rows],0).tolist(),
        'mean_router_entropy_nats':float(np.mean([r['mean_router_entropy_nats'] for r in rows])),
        'mean_fraction_tokens_above_0999_probability':float(np.mean([r['fraction_tokens_above_0999_probability'] for r in rows])),
        'limitation':'Saturation does not alone prove causality or that normalization improves heldout accuracy.',
        'goal_complete':False}
    save_json(root/'expert_saturation_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='by_cell'},indent=2),flush=True)


if __name__=='__main__':main()
