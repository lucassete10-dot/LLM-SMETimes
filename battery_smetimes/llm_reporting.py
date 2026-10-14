"""Descriptive curve summaries with all requested cell/seed denominators."""
import math
import numpy as np


def raw_curve_numeric_metrics(x,y,p):
    x,y,p=[np.asarray(value,dtype=np.float64) for value in (x,y,p)]
    if not (x.ndim==y.ndim==p.ndim==1 and len(x)==len(y)==len(p) and len(x)>=2):raise ValueError('numeric curve coverage')
    if not np.isfinite(np.c_[x,y,p]).all() or np.any(np.diff(x)<=0) or np.any(y<=0):raise ValueError('invalid raw numeric curve')
    error=np.abs(p-y);span=x[-1]-x[0];tail=x>=x[0]+.8*span;drops=[]
    for j in range(3):
        inside=(x>=x[0]+j*span/3)&(x<=x[0]+(j+1)*span/3);xx,yy,pp=x[inside],y[inside],p[inside]
        if len(xx)<3:continue
        width=max(1.,min(5.,(xx[-1]-xx[0])/4));a=xx<=xx[0]+width;b=xx>=xx[-1]-width
        drops.append(100*abs((np.median(yy[a])-np.median(yy[b]))-(np.median(pp[a])-np.median(pp[b]))))
    return {'numeric_evaluable':True,'mape_percent':float(100*np.mean(error/y)),'mae_pp':float(100*error.mean()),
        'tail_mae_pp':float(100*error[tail].mean()),'start_error_pp':float(100*error[0]),'endpoint_error_pp':float(100*error[-1]),
        'stage_drop_max_pp':float(max(drops)) if len(drops)==3 else None,'stage_metrics_available':len(drops)==3}


def finite_metric(row,key):
    value=row.get(key)
    return isinstance(value,(int,float,np.integer,np.floating)) and not isinstance(value,bool) and np.isfinite(value) and value>=0


def summarize_curve_rows(rows,requested,thresholds):
    if requested<len(rows):raise ValueError('more evaluated instances than requested')
    primary=sum(finite_metric(r,'mape_percent') and r['mape_percent']<=thresholds['mape_percent'] for r in rows)
    joint=sum(all(finite_metric(r,k) and r[k]<=limit for k,limit in thresholds.items()) for r in rows)
    distributions={}
    for key in thresholds:
        eligible=[r for r in rows if finite_metric(r,key)];values=np.asarray([r[key] for r in eligible],float)
        worst=sorted(eligible,key=lambda r:(-r[key],r['cell_id'],r['fold'],r['seed']))[:10]
        distributions[key]={'numeric_instances':len(values),'mean':float(values.mean()) if len(values) else None,
            'median':float(np.median(values)) if len(values) else None,'p90':float(np.quantile(values,.9)) if len(values) else None,
            'maximum':float(values.max()) if len(values) else None,
            'worst_decile_mean':float(np.sort(values)[-max(1,math.ceil(len(values)*.1)):].mean()) if len(values) else None,
            'worst_instances':[{k:r[k] for k in ('cell_id','fold','seed')}|{'value':float(r[key])} for r in worst]}
    return {'requested_cell_seed_instances':requested,'evaluated_cell_seed_instances':len(rows),
        'missing_cell_seed_instances':requested-len(rows),'distinct_cells':len({r['cell_id'] for r in rows}),
        'mape_target_passes':int(primary),'mape_target_pass_fraction':primary/requested if requested else None,
        'practical_numeric_passes':int(joint),'practical_numeric_pass_fraction':joint/requested if requested else None,
        'strict_passes':sum(bool(r.get('strict_joint_pass')) for r in rows),
        'metric_distributions':distributions,
        **{k:distributions[k]['mean'] for k in ('mape_percent','mae_pp','tail_mae_pp','stage_drop_max_pp') if k in distributions},
        'mape_failures':[{'cell_id':r['cell_id'],'fold':r['fold'],'seed':r['seed'],'mape_percent':float(r['mape_percent']) if finite_metric(r,'mape_percent') else None,
            'reason':'above target' if finite_metric(r,'mape_percent') else 'numeric metric unavailable'} for r in rows if not finite_metric(r,'mape_percent') or r['mape_percent']>thresholds['mape_percent']]}
