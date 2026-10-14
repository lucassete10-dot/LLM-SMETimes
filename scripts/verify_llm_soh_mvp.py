"""Replay saved LLM adapters, poison future data, and audit query consistency."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import copy,json,argparse
from pathlib import Path
import joblib,numpy as np,torch
from battery_smetimes.research_llm import FrozenSMETimesForecaster,load_adapter,predict_llm,statistics_cards
from battery_smetimes.physical_types import load_registry
from scripts.train_same_type_forecast import load_data
from scripts.prepare_same_type_forecast import save_json,sha


def audit_numeric_lora(bundle,cell,origin):
    """TRAIN-input intervention only; restore adapter before any heldout replay."""
    network=bundle['network']
    if not network.numeric_lora_rank:return None
    trainable={k:p for k,p in network.backbone.named_parameters() if p.requires_grad}
    if not trainable or any(not k.endswith(('.lora_A','.lora_B')) for k in trainable):
        raise ValueError('unexpected trainable pretrained base weights')
    cards=[statistics_cards(cell,origin,network.patch_size,network.history_slots)]
    query=np.array([origin+10,origin+50,origin+100],dtype=float)
    network.eval()
    before=predict_llm(bundle,cell,origin,query)
    text_before=network.encode_cards(cards)
    originals={k:p.detach().clone() for k,p in trainable.items() if k.endswith('.lora_B')}
    try:
        with torch.no_grad():
            for k in originals:trainable[k].add_(.05)
        after=predict_llm(bundle,cell,origin,query)
        text_after=network.encode_cards(cards)
    finally:
        with torch.no_grad():
            for k,value in originals.items():trainable[k].copy_(value)
    restored=predict_llm(bundle,cell,origin,query)
    result={'scope':'first predefined TRAIN cell; in-memory intervention, not predictive quality',
        'train_cell':bundle['processor'].train_ids[0],
        'text_base_encoding_max_change':float((text_before-text_after).abs().max()),
        'numeric_prediction_max_change':float(np.max(abs(before-after))),
        'restoration_max_soh_error':float(np.max(abs(before-restored))),
        'trainable_backbone_parameter_names':list(trainable)}
    result['passed']=bool(result['text_base_encoding_max_change']==0 and
        result['numeric_prediction_max_change']>0 and result['restoration_max_soh_error']<=1e-5)
    if not result['passed']:raise ValueError('numeric adaptation/text cache isolation failed')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
    root=Path(args.run);cfg=json.loads((root/'config.json').read_text())
    audit=Path(cfg['frozen_plan']);frozen=json.loads((audit/'config.json').read_text())
    prepared=json.loads((audit/'audit.json').read_text())['prepared_folder']
    _,cells=load_data(Path(prepared),frozen['config']['integrity_audit'])
    registry=json.loads((audit/'cell_registry.json').read_text())['cells']
    for k,c in cells.items():
        c.update({f:registry[k][f] for f in ('physical_family','reference_established_source_cycle')})
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    outputs=[]
    for candidate in cfg['candidates']:
        dest=root/candidate['name'];processor=joblib.load(dest/'preprocessor.joblib')
        assert processor.train_ids==cfg['plan']['train']
        network=FrozenSMETimesForecaster(cfg['checkpoint'],2*processor.raw_width+1,
            seed=cfg['seed'],dropout=cfg['dropout'],device='cuda',
            normalize_router=candidate.get('normalize_router',False),
            output_head=candidate.get('output_head','point'),pooling=candidate.get('pooling','mean'),
            numeric_dynamics=candidate.get('numeric_dynamics',False),adaptive_fusion=candidate.get('adaptive_fusion',False),
            summary_width=processor.stats_width if candidate.get('summary_encoding',False) else 0,
            residual_experts=candidate.get('residual_experts',False), history_slots=getattr(processor,'history_slots',100),
            numeric_lora_rank=candidate.get('numeric_lora_rank',0))
        load_adapter(network,torch.load(dest/'model.pt',map_location='cpu',weights_only=True))
        bundle={'network':network,'processor':processor,'device':'cuda',
            'registry':load_registry(audit/'source_snapshot/battery_physical_registry.json'),
            'physical_family':cells[processor.train_ids[0]]['physical_family']}
        lora_audit=audit_numeric_lora(bundle,cells[processor.train_ids[0]],cfg['origin'])
        deviations=[];query_errors=[];poison_errors=[]
        with np.load(dest/'heldout_predictions.npz') as saved:
            for cid in cfg['plan']['test']:
                prefix=f'{cid}__{cfg["origin"]}__'
                if prefix+'cycle' not in saved: continue
                x=saved[prefix+'cycle'];before=predict_llm(bundle,cells[cid],cfg['origin'],x)
                deviations.append(float(np.max(abs(before-saved[prefix+'prediction']))))
                chosen=x[[0,len(x)//2,-1]]
                small=predict_llm(bundle,cells[cid],cfg['origin'],chosen)
                query_errors.append(float(np.max(abs(small-before[[0,len(x)//2,-1]]))))
                contaminated=copy.deepcopy(cells[cid])
                contaminated['raw'][contaminated['cycle']>cfg['origin']]=np.inf
                del contaminated['soh'],contaminated['label_cycle']
                poisoned=predict_llm(bundle,contaminated,cfg['origin'],chosen)
                poison_errors.append(float(np.max(abs(poisoned-small))))
                print(candidate['name'],cid,'verified',flush=True)
        result={'candidate':candidate['name'],'cells':len(deviations),'reload_max_soh_error':max(deviations),
            'query_max_soh_error':max(query_errors),'future_poison_max_soh_error':max(poison_errors),
            'tolerance_soh':1e-5,'adapter_sha256':sha(dest/'model.pt'),'numeric_lora_audit':lora_audit}
        result['passed']=bool(max(deviations+query_errors+poison_errors)<=1e-5)
        outputs.append(result);save_json(root/'verification.json',outputs)
        del network,bundle;torch.cuda.empty_cache()
    if not all(r['passed'] for r in outputs):raise ValueError('replay gate failed')


if __name__=='__main__':main()
