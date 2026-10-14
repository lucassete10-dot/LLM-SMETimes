"""Training-only curve-family expressivity vs learned network parameters.

Fits below are diagnostic label fits, never unseen-cell predictions or inputs.
"""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from scipy.optimize import lsq_linear
from threadpoolctl import threadpool_limits
from battery_smetimes.tail_boost import labels
from scripts.train_tail_boost import load_cells
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text())
    assert json.loads((root/'independent_verification.json').read_text())['passed']
    assert sha('battery_smetimes/monotone_network.py')==cfg['source_sha256']['monotone_network.py']
    gaps=json.loads((root/'training_curve_fit_gap.json').read_text())['rows'];reports=[]
    for plan in cfg['plans']:
        cells=load_cells(dict(plan,validation=[]));assert set(cells)==set(plan['train'])
        rows=[]
        for key in plan['train']:
            x,y=labels(cells[key]);take=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int))
            h=(x[take]-100)/100;w=np.linspace(1,3,len(h));w/=w.sum();best=None
            for knee in np.linspace(0,float(h.max()),31):
                hinge=.1*np.logaddexp(0,(h-knee)/.1)
                design=np.column_stack([np.ones(len(h)),-h,-hinge**2])
                result=lsq_linear(design*np.sqrt(w)[:,None],y[take]*np.sqrt(w),bounds=([.5,0,0],[1.25,np.inf,np.inf]))
                assert result.success
                loss=float(np.sum(w*(design@result.x-y[take])**2))
                if best is None or loss<best[0]:best=(loss,knee,result.x)
            loss,knee,values=best;h=(x-100)/100;hinge=.1*np.logaddexp(0,(h-knee)/.1)
            p=values[0]-values[1]*h-values[2]*hinge**2
            rows.append(dict(cell_id=key,label_fit_mape=float(100*np.mean(abs(p-y)/y)),
                training_label_fitted_knee_h=float(knee),anchor=float(values[0]),rate=float(values[1]),
                acceleration=float(values[2]),sampled_weighted_mse=loss))
        network=[]
        for row in gaps:
            if row['fold']==plan['name'] and row['mode']=='network':
                network.append(dict(candidate=row['candidate'],training_mape=row['train']['mape_percent'],
                    median_network_knee_h=float(np.median([v['knee_h'] for v in row['training_predicted_parameters']]))))
        report=dict(fold=plan['name'],training_label_fit_mape=float(np.mean([r['label_fit_mape'] for r in rows])),
            median_training_label_fitted_knee_h=float(np.median([r['training_label_fitted_knee_h'] for r in rows])),
            network=network,cells=rows)
        reports.append(report);print(plan['name']+' diagnostic_train_label_fit_MAPE='+str(report['training_label_fit_mape'])+' median_fitted_knee_h='+str(report['median_training_label_fitted_knee_h']),flush=True)
    save_json(root/'training_parameter_optimization_audit.json',dict(reports=reports,source_sha256=sha(__file__),
        only_training_cells_loaded=True,interpretation='Label-fitted training curves diagnose expressivity/optimization ONLY. Not forecasts, not validation metrics, not model selection, never inference features.',goal_complete=False))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
