"""TRAIN-only weighted loss-gradient audit; not candidate selection."""
import argparse,json
from pathlib import Path
import joblib,numpy as np,torch
from battery_smetimes.research_llm import FrozenSMETimesForecaster,load_adapter,attach_cards
from battery_smetimes.research_sequences import training_contexts,pack_contexts
from battery_smetimes.llm_curve_objective import curve_loss,training_origin_target
from battery_smetimes.llm_knee_supervision import training_knee_target,knee_interval_loss
from battery_smetimes.llm_acceleration_supervision import sustained_acceleration_loss
from battery_smetimes.physical_types import load_registry
from scripts.train_same_type_forecast import load_data
from scripts.prepare_same_type_forecast import save_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    parser.add_argument('--compare-initial-knee',action='store_true');args=parser.parse_args()
    root=Path(args.run);cfg=json.loads((root/'config.json').read_text());audit=Path(cfg['frozen_plan'])
    frozen=json.loads((audit/'config.json').read_text())
    _,cells=load_data(Path(json.loads((audit/'audit.json').read_text())['prepared_folder']),frozen['config']['integrity_audit'])
    registry=json.loads((audit/'cell_registry.json').read_text())['cells']
    for cid,cell in cells.items():cell.update({k:registry[cid][k] for k in ('physical_family','reference_established_source_cycle')})
    torch.set_num_threads(2);records=[]
    for candidate in cfg['candidates']:
        dest=root/candidate['name'];processor=joblib.load(dest/'preprocessor.joblib')
        assert processor.train_ids==cfg['plan']['train']
        contexts=training_contexts(cells,cfg['plan']['train'],processor,cap=cfg['future_targets'])
        order=np.random.default_rng(cfg['seed']+1).permutation(len(contexts))[:cfg['batch_contexts']]
        rows=[contexts[i] for i in order]
        network=FrozenSMETimesForecaster(cfg['checkpoint'],2*processor.raw_width+1,
            seed=cfg['seed'],dropout=cfg['dropout'],device='cuda',
            normalize_router=candidate.get('normalize_router',False),output_head=candidate.get('output_head','point'),
            pooling=candidate.get('pooling','mean'),numeric_dynamics=candidate.get('numeric_dynamics',False),
            adaptive_fusion=candidate.get('adaptive_fusion',False),
            summary_width=processor.stats_width if candidate.get('summary_encoding',False) else 0,
            residual_experts=candidate.get('residual_experts',False), history_slots=getattr(processor,'history_slots',100),
            initial_knee_bias=candidate.get('initial_knee_bias',1.),numeric_lora_rank=candidate.get('numeric_lora_rank',0))
        network.bind_query_scaler(processor.query_scale.scaler)
        bundle={'network':network,'processor':processor,'registry':load_registry(audit/'source_snapshot/battery_physical_registry.json'),
            'physical_family':cells[rows[0]['cell_id']]['physical_family'],'device':'cuda'}
        attach_cards(bundle,rows,cells)
        batch=pack_contexts(rows,'cuda');cards=torch.stack([r['card_embeddings'] for r in rows]).to('cuda')
        inverse=processor.query_scale.scaler.inverse_transform(batch['query'].cpu().numpy().reshape(-1,3))
        cycles=torch.as_tensor(inverse[:,0].reshape(len(rows),-1)*100,device='cuda',dtype=torch.float32)
        states=('initial','initial_early_knee','selected') if args.compare_initial_knee else ('initial','selected')
        for state in states:
            if state=='initial_early_knee':
                if network.output_head!='integrated_rate':raise ValueError('knee diagnostic requires integrated head')
                with torch.no_grad():network.head[-1].bias[3].fill_(-1.)
            if state=='selected':load_adapter(network,torch.load(dest/'model.pt',map_location='cpu',weights_only=True))
            network.train();torch.manual_seed(cfg['seed']+777)
            head_outputs=[]
            hook=network.head.register_forward_hook(lambda module,inputs,output: head_outputs.append(output))
            encoding=network.encode(batch['history'],batch['length'],cards,
                batch['statistics'] if network.summary_width else None)
            prediction=network.decode(encoding,batch['conditions'],batch['query'])
            individual,terms=curve_loss(prediction,batch['truth'],cycles,batch['valid'],candidate['stage_weight'],candidate['tail_weight'],candidate.get('rate_weight',0.))
            acceleration_auxiliary=None
            if candidate.get('acceleration_weight',0):
                acceleration_auxiliary,_=sustained_acceleration_loss(prediction,batch['truth'],cycles,batch['valid'])
                individual=individual+candidate['acceleration_weight']*acceleration_auxiliary
            knee_auxiliary=None
            if candidate.get('knee_weight',0):
                targets=[training_knee_target(cells[r['cell_id']],r['origin']) for r in rows]
                lower,upper,valid=[torch.as_tensor([t[k] for t in targets],device='cuda',dtype=torch.float32)
                                   for k in ('lower','upper','valid')]
                knee_auxiliary=knee_interval_loss(head_outputs[0],lower,upper,valid)
                individual=individual+candidate['knee_weight']*knee_auxiliary
            if candidate.get('start_weight',0):
                queries=processor.query_scale.transform(np.array([[r['origin']/100,0.,0.] for r in rows]))
                start=network.decode(encoding,batch['conditions'],torch.as_tensor(queries[:,None],device='cuda'))[:,0]
                targets=[training_origin_target(cells[r['cell_id']],r['origin']) for r in rows]
                target=torch.as_tensor([t[0] for t in targets],device='cuda');valid=torch.as_tensor([t[1] for t in targets],device='cuda')
                terms['start_aux']=(start-target).square()*valid
                individual=individual+candidate['start_weight']*terms['start_aux']
            weighted={'forecast':(individual*batch['weight']).mean(),'balance':cfg['balance_coefficient']*network.balance_loss}
            if knee_auxiliary is not None:
                weighted['knee_aux_weighted']=(candidate['knee_weight']*knee_auxiliary*batch['weight']).mean()
            if acceleration_auxiliary is not None:
                weighted['acceleration_aux_weighted']=(candidate['acceleration_weight']*acceleration_auxiliary*batch['weight']).mean()
            hook.remove()
            decoder_diagnostic=None
            if network.output_head=='integrated_rate':
                parameters=head_outputs[0]
                gradient=torch.autograd.grad(weighted['forecast'],parameters,retain_graph=True)[0]
                names=('level','base_rate','acceleration','knee','transition_width','recovery','recovery_time')
                decoder_diagnostic={name:{'raw_parameter_by_train_cell':parameters[:,i].detach().cpu().tolist(),
                    'forecast_gradient_by_train_cell':gradient[:,i].detach().cpu().tolist(),
                    'gradient_norm':float(gradient[:,i].detach().norm())} for i,name in enumerate(names)}
            selected=[(name,p) for name,p in network.named_parameters() if p.requires_grad]
            norms={}
            for name,loss in weighted.items():
                grads=torch.autograd.grad(loss,[p for _,p in selected],retain_graph=True,allow_unused=True)
                norms[name]={key:float(g.detach().norm()) if g is not None else 0. for (key,_),g in zip(selected,grads)}
            records.append({'candidate':candidate['name'],'state':state,'train_cells':[r['cell_id'] for r in rows],
                'weighted_losses':{k:float(v.detach()) for k,v in weighted.items()},'gradient_norms':norms,
                'decoder_parameter_diagnostic':decoder_diagnostic,
                'scope':'fixed first epoch TRAIN minibatch; reset diagnostic dropout seed; initial and saved selected adapter; not exact historical gradients'})
            print(candidate['name'],state,records[-1]['weighted_losses'],
                'projection gradient',{k:norms[k]['numeric_projection.0.weight'] for k in norms},flush=True)
        del network,bundle;torch.cuda.empty_cache()
    filename='initial_knee_gradient_diagnostic.json' if args.compare_initial_knee else 'loss_balance_diagnostic.json'
    save_json(root/filename,records)


if __name__=='__main__':main()
