"""Training-only fit capacity vs learned signal-to-parameter mapping, not a forecast claim."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.parameter_ridge import predict,trajectory,parameters_from,FIELDS
from battery_smetimes.conditional_drift import encode
from battery_smetimes.curve_fidelity import curve_metrics
from battery_smetimes.tail_boost import labels
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_tail_boost import load_cells
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text());reports=[]
    assert json.loads((root/'progress.json').read_text())['training_complete']
    for plan in cfg['plans']:
        cells=load_cells(dict(plan,validation=[]));assert set(cells)==set(plan['train'])
        fits=json.loads((root/(plan['name']+'_training_parameters.json')).read_text());oracle=[]
        for key,row in zip(plan['train'],fits):
            assert row['cell_id']==key;x,y=labels(cells[key]);p=trajectory([row[f] for f in FIELDS],x)
            oracle.append(dict(cell_id=key,**curve_metrics(x,y,p)))
        validation_rows=json.loads((root/'summary.json').read_text())
        for index,params in enumerate(cfg['candidates']):
            if params is None:continue
            folder=root/plan['name']/f'candidate_{index}';b=joblib.load(folder/'model.joblib');rows=[];arrays={};predicted_parameters=[]
            for key in plan['train']:
                cc=clean(cells[key]);x,y=labels(cells[key]);p=predict(b,cc,x)
                rows.append(dict(cell_id=key,**curve_metrics(x,y,p)))
                xx=encode(b['encoder'],cc)[None];u=b['model'].predict(xx)[0]*b['target_scale']+b['target_mean']
                predicted_parameters.append(parameters_from(u,b['logarithmic']).tolist())
                arrays.update({key+'__100__'+f:a for f,a in [('cycle',x),('truth',y),('prediction',p)]})
            np.savez_compressed(folder/'training_diagnostic_predictions.npz',**arrays)
            save_json(folder/'training_diagnostic_metrics.json',rows)
            val=next(r for r in validation_rows if r['fold']==plan['name'] and r['candidate']==index)
            report=dict(fold=plan['name'],candidate=index,parameters=params,training_cells=len(rows),oracle_training_mape=float(np.mean([r['mape_percent'] for r in oracle])),learned_training_mape=float(np.mean([r['mape_percent'] for r in rows])),validation_mape=val['mape_percent'],training_parameter_medians=np.median([[r[f] for f in FIELDS] for r in fits],axis=0).tolist(),predicted_training_parameter_medians=np.median(predicted_parameters,axis=0).tolist(),model_sha256=sha(folder/'model.joblib'))
            reports.append(report);print(json.dumps(report),flush=True)
    save_json(root/'training_mapping_gap.json',dict(reports=reports,loaded_train_only=True,oracle_fits_are_not_forecasts=True,goal_complete=False,source_sha256=sha(__file__)))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
