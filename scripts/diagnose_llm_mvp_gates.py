"""Train-side activation evidence before changing the SMETimes expert router."""
import argparse,json
from pathlib import Path
import joblib,numpy as np,torch
from battery_smetimes.research_llm import FrozenSMETimesForecaster,load_adapter,predict_llm
from battery_smetimes.physical_types import load_registry
from scripts.train_same_type_forecast import load_data
from scripts.prepare_same_type_forecast import save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
    root=Path(args.run);cfg=json.loads((root/'config.json').read_text());audit=Path(cfg['frozen_plan'])
    frozen=json.loads((audit/'config.json').read_text());prepared=json.loads((audit/'audit.json').read_text())['prepared_folder']
    _,cells=load_data(Path(prepared),frozen['config']['integrity_audit'])
    records=json.loads((audit/'cell_registry.json').read_text())['cells']
    for k,c in cells.items():c.update({f:records[k][f] for f in ('physical_family','reference_established_source_cycle')})
    torch.set_num_threads(2)
    results=[]
    for candidate in cfg['candidates']:
        dest=root/candidate['name'];processor=joblib.load(dest/'preprocessor.joblib')
        network=FrozenSMETimesForecaster(cfg['checkpoint'],2*processor.raw_width+1,
            seed=cfg['seed'],dropout=cfg['dropout'],device='cuda',
            normalize_router=candidate.get('normalize_router',False),
            output_head=candidate.get('output_head','point'),pooling=candidate.get('pooling','mean'),
            numeric_dynamics=candidate.get('numeric_dynamics',False),adaptive_fusion=candidate.get('adaptive_fusion',False),
            summary_width=processor.stats_width if candidate.get('summary_encoding',False) else 0,
            residual_experts=candidate.get('residual_experts',False), history_slots=getattr(processor,'history_slots',100),
            numeric_lora_rank=candidate.get('numeric_lora_rank',0))
        load_adapter(network,torch.load(dest/'model.pt',map_location='cpu',weights_only=True))
        observed=[]
        def hook(module,inputs,output):
            h=inputs[0].detach().float();g=output.detach().float().softmax(-1)
            observed.append({'hidden_rms':float(h.square().mean().sqrt()),
                'logit_abs_max':float(output.detach().abs().max()),'mean_max_probability':float(g.max(-1).values.mean()),
                'entropy':float(-(g*g.clamp_min(1e-12).log()).sum(-1).mean()),
                'usage':g.mean((0,1)).cpu().tolist()})
        handle=network.gate.register_forward_hook(hook)
        fusion=[]
        def fusion_hook(module,inputs,output):
            alpha=torch.sigmoid(output.detach().float()+network.fusion_logit.detach())
            fusion.append({'numeric_weight_mean':float(alpha.mean()),'numeric_weight_min':float(alpha.min()),
                'numeric_weight_max':float(alpha.max()),'saturated_fraction':float(((alpha<.05)|(alpha>.95)).float().mean())})
        fusion_handle=network.fusion_gate.register_forward_hook(fusion_hook) if network.adaptive_fusion else None
        bundle={'network':network,'processor':processor,'device':'cuda',
            'registry':load_registry(audit/'source_snapshot/battery_physical_registry.json'),
            'physical_family':cells[processor.train_ids[0]]['physical_family']}
        forecasts=[]
        for cid in cfg['plan']['train'][:12]:
            origin=cfg['origin']
            forecasts.append(predict_llm(bundle,cells[cid],origin,origin+np.array([1.,100.,300.,700.])))
        handle.remove()
        if fusion_handle is not None:fusion_handle.remove()
        result={'candidate':candidate['name'],'scope':'first 12 pre-fixed TRAIN cells; no heldout selection',
            'cells':cfg['plan']['train'][:12],'activations':observed,
            'forecast_between_cell_std_soh':np.std(forecasts,axis=0).tolist(),
            'queries':(cfg['origin']+np.array([1,100,300,700])).tolist(),
            'fusion_activations':fusion}
        results.append(result);save_json(root/'gate_diagnostic.json',results)
        print(candidate['name'],'mean router max probability',np.mean([r['mean_max_probability'] for r in observed]),
            'entropy',np.mean([r['entropy'] for r in observed]),'hidden RMS',np.mean([r['hidden_rms'] for r in observed]),flush=True)
        del network,bundle;torch.cuda.empty_cache()


if __name__=='__main__':main()
