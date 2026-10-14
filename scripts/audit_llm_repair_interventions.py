"""Visible-input interventions; labels are stripped before model inference."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,copy,json
from pathlib import Path
import joblib,numpy as np,torch
from battery_smetimes.research_llm import load_adapter,predict_llm
from battery_smetimes.physical_types import load_registry
from scripts.audit_llm_curve_repair import load_frozen,fingerprint
from scripts.train_llm_curve_repair import create_network
from scripts.prepare_same_type_forecast import save_json


def visible_only(cell,origin):
    result=copy.deepcopy(cell)
    for key in ('soh','label_cycle','unfiltered_soh','capacity','eol','lifetime'):result.pop(key,None)
    take=result['cycle']<=origin
    result['cycle']=result['cycle'][take];result['raw']=result['raw'][take]
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--roles',nargs='+',choices=['training','validation'],default=['training']);args=ap.parse_args()
    root=args.run;cfg=json.loads((root/'config.json').read_text());_,cells,plan,registry,_=load_frozen(cfg)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    processor=joblib.load(root/'preprocessor.joblib');network=create_network(cfg,processor,cfg['architecture'],cfg['dropout'])
    load_adapter(network,torch.load(root/'model.pt',map_location='cpu',weights_only=True));network.eval()
    bundle={'network':network,'processor':processor,'device':'cuda','registry':load_registry(registry),
        'physical_family':cells[cfg['train_ids'][0]]['physical_family']}
    origin=cfg['origin'];query=origin+np.asarray([1.,10.,50.,100.,300.,600.,1000.])
    rows=[];arrays={};rng=np.random.default_rng(cfg['seed']+107)
    for role in args.roles:
        ids=cfg['train_ids'] if role=='training' else plan['validation']
        for index,cid in enumerate(ids):
            cell=visible_only(cells[cid],origin);donor=visible_only(cells[ids[(index+1)%len(ids)]],origin)
            before=predict_llm(bundle,cell,origin,query);variants={}
            shuffled=copy.deepcopy(cell);shuffled['raw']=shuffled['raw'][rng.permutation(len(shuffled['raw']))]
            variants['history_order_shuffled']=shuffled
            swapped=copy.deepcopy(cell);swapped['raw']=donor['raw'].copy();swapped['cycle']=donor['cycle'].copy()
            variants['history_from_next_cell']=swapped
            changed=copy.deepcopy(cell);changed['metadata']=copy.deepcopy(donor['metadata'])
            variants['known_conditions_from_next_cell']=changed
            arrays[cid+'__baseline']=before
            for name,value in variants.items():
                after=predict_llm(bundle,value,origin,query)
                a,b=processor.model_input(cell,origin),processor.model_input(value,origin)
                input_changed=any(a[k].shape!=b[k].shape or not np.array_equal(a[k],b[k]) for k in ('history','statistics','conditions'))
                rows.append({'role':role,'cell':cid,'donor':ids[(index+1)%len(ids)],'intervention':name,
                    'numeric_input_changed':input_changed,'max_prediction_change_pp':float(100*np.max(abs(after-before))),
                    'mean_prediction_change_pp':float(100*np.mean(abs(after-before)))})
                arrays[cid+'__'+name]=after
    arrays['fixed_query_source_cycles']=query
    np.savez_compressed(root/'visible_input_interventions.npz',**arrays)
    report={'scope':'Visible-input dependence only; no target-based queries, accuracy estimates or candidate selection',
        'loader_limitation':'Legacy loader reads prepared labels for integrity; labels are stripped before inference and do not choose queries or interventions.',
        'model_sha256':fingerprint(root/'model.pt'),'source_sha256':fingerprint(Path(__file__)),
        'roles':args.roles,'fixed_queries_do_not_use_true_endpoints':True,'per_cell':rows,
        'summary':{name:{'changed_input_cases':sum(r['numeric_input_changed'] for r in rows if r['intervention']==name),
            'mean_max_prediction_change_pp':float(np.mean([r['max_prediction_change_pp'] for r in rows if r['intervention']==name]))}
            for name in variants},'limitation':'Changed output establishes dependence, not accuracy benefit or causality.',
        'goal_complete':False}
    save_json(root/'visible_input_interventions.json',report);print(json.dumps(report['summary'],indent=2),flush=True)


if __name__=='__main__':main()
