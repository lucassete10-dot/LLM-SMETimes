"""TRAIN-prefix-only decoder parameter variation; never target coefficient fitting."""
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
    rows=[];arrays={};residuals=[]
    with torch.inference_mode():
        for cid in cfg['train_ids']:
            v=processor.model_input(cells[cid],cfg['origin']);embedding=cards_embedding(bundle,cells[cid],cfg['origin'])[None].cuda()
            enc=net.encode(torch.as_tensor(v['history'][None],device='cuda'),torch.tensor([v['length']],device='cuda'),embedding)
            q=torch.as_tensor(processor.future_query(cfg['origin'],np.asarray([cfg['origin']+1]))[None],device='cuda')
            _,p=net.decode(enc,torch.as_tensor(v['conditions'][None],device='cuda'),q,return_parameters=True)
            local=p['local_recovery_values'][0].cpu().numpy();rates=p['rates'][0].cpu().numpy()
            power=abs(np.fft.rfft(local-local.mean()))**2;freq=np.fft.rfftfreq(len(local),d=cfg['local_recovery_spacing_cycles'])
            rows.append({'cell':cid,'origin_soh':float(p['origin_soh'][0]),'rate_min':float(rates.min()),'rate_max':float(rates.max()),
                'local_min_pp':float(100*local.min()),'local_max_pp':float(100*local.max()),
                'mean_adjacent_local_change_pp':float(100*np.mean(abs(np.diff(local)))),
                'fraction_local_nodes_above_095_bound':float(np.mean(abs(local)>.95*cfg['local_recovery_max_soh'])),
                'local_power_period_below_20_cycles_fraction':float(power[freq>1/20].sum()/max(power.sum(),1e-30))})
            arrays[cid+'__local']=local;arrays[cid+'__rates']=rates;residuals.append(local)
    singular=np.linalg.svd(np.asarray(residuals)-np.mean(residuals,axis=0),compute_uv=False)
    report={'scope':'TRAIN visible-prefix-only model parameters on fixed nodes; no target-based time grid or accuracy scoring',
        'loader_limitation':'Legacy integrity loader reads prepared labels; none enter inference or this parameter audit',
        'model_sha256':fingerprint(root/'model.pt'),'source_sha256':fingerprint(Path(__file__)),
        'fixed_local_node_source_cycles':(cfg['origin']+100*net.local_recovery_knots.cpu().numpy()).tolist(),
        'centered_local_singular_values':singular.tolist(),'by_cell':rows,'goal_complete':False}
    np.savez_compressed(root/'decoder_parameters.npz',**arrays);save_json(root/'decoder_parameter_audit.json',report)
    print(json.dumps({'by_cell':rows,'singular_values':singular.tolist()},indent=2),flush=True)


if __name__=='__main__':main()
