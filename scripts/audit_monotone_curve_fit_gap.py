"""Report training-vs-validation fit for ALL candidates, never select/exclude."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
import torch
from battery_smetimes.monotone_network import predict,CurveNet
from battery_smetimes.conditional_drift import prefix
from threadpoolctl import threadpool_limits
from battery_smetimes.tail_boost import labels
from battery_smetimes.curve_fidelity import curve_metrics
from scripts.train_tail_boost import load_cells
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text())
    assert json.loads((root/'independent_verification.json').read_text())['passed']
    for file in ('battery_smetimes/monotone_network.py','battery_smetimes/conditional_drift.py'):
        assert sha(file)==cfg['source_sha256'][Path(file).name]
    rows=[]
    for plan in cfg['plans']:
        cells=load_cells(plan);assert not set(cells)&set(plan['test'])
        for index in range(len(cfg['candidates'])):
            folder=root/plan['name']/f'candidate_{index}';b=joblib.load(folder/'model.joblib')
            assert b['train_ids']==plan['train']
            train=[]
            for key in plan['train']:
                x,y=labels(cells[key]);p=predict(b,clean(cells[key]),x)
                train.append(dict(cell_id=key,**curve_metrics(x,y,p)))
            val=json.loads((folder/'validation_metrics.json').read_text());assert {r['cell_id'] for r in val}==set(plan['validation'])
            fields=('mape_percent','late_mape','endpoint_pp','drop_error_pp')
            train_mean={f:float(np.mean([r[f] for r in train])) for f in fields}
            val_mean={f:float(np.mean([r[f] for r in val])) for f in fields}
            row=dict(fold=plan['name'],candidate=index,mode=b['mode'],train=train_mean,validation=val_mean,training_cells=train)
            if b['mode']=='network':
                z=b['scaler'].transform(b['imputer'].transform(np.stack([prefix(cells[k]) for k in plan['train']])))
                net=CurveNet(b['input_width'],b['width']);net.load_state_dict({k:torch.from_numpy(v) for k,v in b['state'].items()})
                with torch.no_grad():params=net.parameters_for(torch.tensor(z,dtype=torch.float32)).numpy()
                row['training_predicted_parameters']=[dict(cell_id=k,anchor=float(v[0]),rate=float(v[1]),acceleration=float(v[2]),knee_h=float(v[3])) for k,v in zip(plan['train'],params)]
                row['last50_loss_mean']=float(np.mean(b['history'][-50:]))
                row['previous50_loss_mean']=float(np.mean(b['history'][-100:-50]))
                print(plan['name']+' candidate='+str(index)+' tau_h_min_median_max='+str(np.quantile(params[:,3],[0,.5,1]).tolist()),flush=True)
            rows.append(row);print(plan['name']+' candidate='+str(index)+' trainMAPE='+str(round(train_mean['mape_percent'],3))+' valMAPE='+str(round(val_mean['mape_percent'],3)),flush=True)
    save_json(root/'training_curve_fit_gap.json',dict(rows=rows,source_sha256=sha(__file__),
        interpretation='Training fit is diagnostic only, not unseen-cell accuracy. All candidates retained; no model choice or exclusions.',outer_loaded=False,goal_complete=False))


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
