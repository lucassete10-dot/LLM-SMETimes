"""Fixed TRAIN-only counterfactuals: does the LLM battery history matter?"""
import argparse,json
from pathlib import Path
import joblib,numpy as np,torch
from battery_smetimes.research_llm import FrozenSMETimesForecaster,load_adapter,cards_embedding
from battery_smetimes.physical_types import load_registry
from scripts.train_same_type_forecast import load_data
from scripts.prepare_same_type_forecast import save_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    root=Path(args.run);cfg=json.loads((root/'config.json').read_text())
    audit=Path(cfg['frozen_plan']);frozen=json.loads((audit/'config.json').read_text())
    prepared=json.loads((audit/'audit.json').read_text())['prepared_folder']
    _,cells=load_data(Path(prepared),frozen['config']['integrity_audit'])
    registry=json.loads((audit/'cell_registry.json').read_text())['cells']
    for k,c in cells.items():c.update({f:registry[k][f] for f in ('physical_family','reference_established_source_cycle')})
    torch.set_num_threads(2);outputs=[]
    # Predefined TRAIN order, never selected by future SOH or error.
    ids=cfg['plan']['train'][:12]
    for candidate in cfg['candidates']:
        dest=root/candidate['name'];processor=joblib.load(dest/'preprocessor.joblib')
        network=FrozenSMETimesForecaster(cfg['checkpoint'],2*processor.raw_width+1,
            seed=cfg['seed'],dropout=cfg['dropout'],device='cuda',
            normalize_router=candidate.get('normalize_router',False),output_head=candidate.get('output_head','point'),
            pooling=candidate.get('pooling','mean'),numeric_dynamics=candidate.get('numeric_dynamics',False),
            adaptive_fusion=candidate.get('adaptive_fusion',False),
            summary_width=processor.stats_width if candidate.get('summary_encoding',False) else 0,
            residual_experts=candidate.get('residual_experts',False), history_slots=getattr(processor,'history_slots',100),
            numeric_lora_rank=candidate.get('numeric_lora_rank',0))
        load_adapter(network,torch.load(dest/'model.pt',map_location='cpu',weights_only=True));network.eval()
        bundle={'network':network,'processor':processor,'device':'cuda',
            'registry':load_registry(audit/'source_snapshot/battery_physical_registry.json'),
            'physical_family':cells[ids[0]]['physical_family']}
        encodings={};conditions={};trajectories={};records=[];pooling_weights={};pool_scores=[];dynamics_effects={};summary_effects={}
        pool_handle = (network.pool_score.register_forward_hook(
            lambda module, inputs, output: pool_scores.append(output.detach()))
            if network.pooling == 'attention' else None)
        future=cfg['origin']+np.array([1.,50.,100.,200.,400.,800.])
        query=torch.as_tensor(processor.future_query(cfg['origin'],future)[None],device='cuda')
        with torch.inference_mode():
            for cid in ids:
                value=processor.model_input(cells[cid],cfg['origin'])
                encodings[cid]=network.encode(torch.as_tensor(value['history'][None],device='cuda'),
                    torch.as_tensor([value['length']],device='cuda'),cards_embedding(bundle,cells[cid],cfg['origin'])[None].to('cuda'),
                    torch.as_tensor(value['statistics'][None],device='cuda') if network.summary_width else None)
                if pool_handle is not None:
                    count=(value['length']+network.patch_size-1)//network.patch_size
                    probabilities=torch.softmax(pool_scores.pop()[0,:count,0],dim=0)
                    pooling_weights[cid]={'patch_weights':probabilities.cpu().tolist(),
                        'effective_patch_count':float(torch.exp(-(probabilities*probabilities.clamp_min(1e-12).log()).sum())),
                        'maximum_patch_weight':float(probabilities.max())}
                conditions[cid]=torch.as_tensor(value['conditions'][None],device='cuda')
                trajectories[cid]=network.decode(encodings[cid],conditions[cid],query)
                if network.numeric_dynamics:
                    # Zero measurement rates ONLY, retain their availability
                    # and actual intervals and the untouched numeric/card path.
                    # Hook does not inspect labels or alter saved weights.
                    def mask_rates(module,inputs):
                        value=inputs[0].clone().reshape(1,network.history_slots,2*processor.raw_width+1)
                        value[...,:processor.raw_width]=0
                        return (value.reshape(1,network.history_slots//network.patch_size,-1),)
                    handle=network.dynamics_projection.register_forward_pre_hook(mask_rates)
                    try:
                        masked_encoding=network.encode(torch.as_tensor(value['history'][None],device='cuda'),
                            torch.as_tensor([value['length']],device='cuda'),cards_embedding(bundle,cells[cid],cfg['origin'])[None].to('cuda'),
                            torch.as_tensor(value['statistics'][None],device='cuda') if network.summary_width else None)
                    finally:handle.remove()
                    masked_curve=network.decode(masked_encoding,conditions[cid],query)
                    dynamics_effects[cid]=(100*(masked_curve-trajectories[cid]).abs()).cpu().numpy()[0].tolist()
                if network.summary_width:
                    # Mask standardized signal summary fields only; keep the
                    # last nine time/known-condition fields and raw/card path.
                    def mask_summary_signals(module,inputs):
                        summary=inputs[0].clone();summary[...,:-9]=0
                        return (summary,)
                    handle=network.summary_projection.register_forward_pre_hook(mask_summary_signals)
                    try:
                        masked_encoding=network.encode(torch.as_tensor(value['history'][None],device='cuda'),
                            torch.as_tensor([value['length']],device='cuda'),cards_embedding(bundle,cells[cid],cfg['origin'])[None].to('cuda'),
                            torch.as_tensor(value['statistics'][None],device='cuda'))
                    finally:handle.remove()
                    masked_curve=network.decode(masked_encoding,conditions[cid],query)
                    summary_effects[cid]=(100*(masked_curve-trajectories[cid]).abs()).cpu().numpy()[0].tolist()
            for i,cid in enumerate(ids):
                donor=ids[(i+1)%len(ids)]
                assert cells[cid]['physical_family']==cells[donor]['physical_family']
                # Retain recipient conditions/query, exchange only LLM-derived
                # historical representation. This is a counterfactual, not a
                # valid prediction or model selection score.
                swapped=network.decode(encodings[donor],conditions[cid],query)
                masked=network.decode(torch.zeros_like(encodings[cid]),conditions[cid],query)
                record={'cell':cid,'donor':donor,
                    'history_swap_abs_delta_pp':(100*(swapped-trajectories[cid]).abs()).cpu().numpy()[0].tolist(),
                    'llm_encoding_zero_abs_delta_pp':(100*(masked-trajectories[cid]).abs()).cpu().numpy()[0].tolist(),
                    'encoding_rms':float(encodings[cid].square().mean().sqrt())}
                if network.output_head=='integrated_rate':
                    record['raw_trajectory_parameters']=network.head(torch.cat((encodings[cid],conditions[cid]),-1)).cpu().numpy()[0].tolist()
                if cid in pooling_weights:
                    record['attention_pooling']=pooling_weights[cid]
                if cid in dynamics_effects:
                    record['measurement_rates_zero_abs_delta_pp']=dynamics_effects[cid]
                if cid in summary_effects:
                    record['standardized_signal_summary_zero_abs_delta_pp']=summary_effects[cid]
                records.append(record)
        if pool_handle is not None:pool_handle.remove()
        outputs.append({'candidate':candidate['name'],'scope':'first12 frozen TRAIN ids; no labels used',
            'counterfactual_only':True,'query_source_cycles':future.tolist(),'cells':records,
            'mean_swap_delta_pp':np.mean([r['history_swap_abs_delta_pp'] for r in records],axis=0).tolist(),
            'mean_zero_encoding_delta_pp':np.mean([r['llm_encoding_zero_abs_delta_pp'] for r in records],axis=0).tolist(),
            'forecast_between_cell_std_pp':(100*np.std([trajectories[k].cpu().numpy()[0] for k in ids],axis=0)).tolist()})
        if dynamics_effects:
            outputs[-1]['mean_measurement_rates_zero_delta_pp']=np.mean(list(dynamics_effects.values()),axis=0).tolist()
        if summary_effects:
            outputs[-1]['mean_standardized_signal_summary_zero_delta_pp']=np.mean(list(summary_effects.values()),axis=0).tolist()
        save_json(root/'history_effect_diagnostic.json',outputs)
        print(candidate['name'],'swap delta pp',outputs[-1]['mean_swap_delta_pp'],flush=True)
        del network,bundle;torch.cuda.empty_cache()


if __name__=='__main__':main()
