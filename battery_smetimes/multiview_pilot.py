"""Registered lightweight MIT multiview pilot, not a paper reproduction."""
import copy
import time
import warnings
import numpy as np
import torch
from torch import nn
from battery_smetimes.mit_segment_features import prefix, columns, FEATURE_NAMES
from battery_smetimes.curve_fidelity import targets, curve_metrics, selection_score


def views(cell):
    vector = prefix(cell, dict(block='VIT', context='segments'))
    width = len(columns('VIT'))
    assert len(vector) == 11*width+13
    return [vector[:2*width], vector[2*width:11*width+4], vector[-9:]]


def manifest():
    width = len(columns('VIT'))
    return dict(raw_columns=[FEATURE_NAMES[i] for i in columns('VIT')],
                view_dimensions=[2*width, 9*width+4, 9],
                roles=['initial_and_recent_state', 'differences_slopes_quarter_history', 'coverage_and_known_conditions'],
                source_cycle_limit=100, horizon_features=['(cycle-100)/100','log1p(h)','h_squared'])


class CurveNet(nn.Module):
    def __init__(self, dimensions, mode):
        super().__init__()
        if mode not in ('concat','gate'):raise ValueError('unregistered mode')
        self.mode = mode
        self.encoders = nn.ModuleList([nn.Sequential(nn.Linear(d,16),nn.Tanh()) for d in dimensions])
        self.gate = nn.Linear(48,3) if mode=='gate' else None
        self.head = nn.Sequential(nn.Linear(51,32), nn.Tanh(), nn.Linear(32,1))

    def forward(self, xs, horizon):
        z = [encoder(x) for encoder,x in zip(self.encoders,xs)]
        joined = torch.cat(z,dim=1)
        if self.gate is not None:
            weights = torch.softmax(self.gate(joined),dim=1)
            joined = torch.cat([v*weights[:,i:i+1] for i,v in enumerate(z)],dim=1)
        return 1+.1*self.head(torch.cat([joined,horizon],dim=1)).squeeze(1)


def fit_preprocessing(cells, ids):
    stacked = [np.stack([views(cells[k])[i] for k in ids]) for i in range(3)]
    result = []
    for x in stacked:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            median = np.nanmedian(x,axis=0)
        empty = ~np.isfinite(median)
        median[empty]=0.
        filled=np.where(np.isnan(x),median,x)
        if not np.isfinite(filled).all():raise ValueError('invalid train inputs')
        result.append(dict(median=median, mean=filled.mean(axis=0),
                           scale=np.maximum(filled.std(axis=0),1e-6), empty_columns=np.flatnonzero(empty).tolist()))
    return result


def tensors(bundle, cell, future):
    if cell['physical_family']!=bundle['family']:raise ValueError('incompatible family')
    f=np.asarray(future,dtype=float)
    if f.ndim!=1 or len(f)==0 or not np.isfinite(f).all() or np.any(f<=100):raise ValueError('invalid query')
    xs=[]
    for x,s in zip(views(cell),bundle['preprocessing']):
        x=np.where(np.isnan(x),s['median'],x)
        z=(x-s['mean'])/s['scale']
        if not np.isfinite(z).all():raise ValueError('invalid signal')
        xs.append(torch.from_numpy(np.tile(z.astype('float32'),(len(f),1))))
    h=(f-100)/100
    return xs,torch.tensor(np.column_stack([h,np.log1p(h),h*h]),dtype=torch.float32)


def predict(bundle, cell, future):
    xs,h=tensors(bundle,cell,future)
    bundle['model'].eval()
    with torch.no_grad():return bundle['model'](xs,h).numpy().astype(float)


def evaluate(bundle,cells,ids):
    rows,arrays=[],{}
    for key in ids:
        c,y=targets(cells[key]);p=predict(bundle,cells[key],c)
        e=p-y;ss=float(np.sum((y-y.mean())**2))
        row=dict(cell_id=key,**curve_metrics(c,y,p))
        row.update(mae_soh=float(np.mean(abs(e))),r2=1-float(e@e)/ss if ss else None,
                   r2_undefined=not bool(ss),start_error_pp=float(100*abs(e[0])))
        rows.append(row)
        arrays.update({key+'__100__'+n:v for n,v in [('cycle',c),('truth',y),('prediction',p)]})
    return rows,arrays


def fit(cells,train_ids,validation_ids,seed,mode,epochs):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(seed)
    assert not set(train_ids)&set(validation_ids)
    family=cells[train_ids[0]]['physical_family']
    assert all(cells[k]['physical_family']==family for k in train_ids+validation_ids)
    bundle=dict(family=family,train_ids=list(train_ids),validation_ids=list(validation_ids),seed=seed,mode=mode,
                preprocessing=fit_preprocessing(cells,train_ids),input_manifest=manifest())
    net=CurveNet(manifest()['view_dimensions'],mode);bundle['model']=net
    chunks=[];ys=[];ws=[]
    for key in train_ids:
        c,y=targets(cells[key]);idx=np.unique(np.linspace(0,len(c)-1,min(96,len(c))).round().astype(int))
        chunks.append(tensors(bundle,cells[key],c[idx]));ys.append(y[idx])
        w=np.linspace(1,3,len(idx));ws.append(w/w.sum())
    xs=[torch.cat([chunk[0][i] for chunk in chunks]) for i in range(3)]
    h=torch.cat([chunk[1] for chunk in chunks]);y=torch.tensor(np.concatenate(ys),dtype=torch.float32)
    w=torch.tensor(np.concatenate(ws),dtype=torch.float32);w=w/w.sum()
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
    best,best_state,best_epoch=float('inf'),None,None;history=[];start=time.perf_counter()
    for epoch in range(1,epochs+1):
        net.train();optimizer.zero_grad()
        loss=(torch.nn.functional.smooth_l1_loss(net(xs,h),y,reduction='none',beta=.02)*w).sum()
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        loss.backward();optimizer.step()
        if epoch%10==0 or epoch==epochs:
            rows,_=evaluate(bundle,cells,validation_ids);score=selection_score(rows)
            history.append(dict(epoch=epoch,train_loss=float(loss.detach()),validation_score=score))
            if score<best:best,best_epoch,best_state=score,epoch,copy.deepcopy(net.state_dict())
    net.load_state_dict(best_state);net.eval()
    bundle.update(fit_seconds=time.perf_counter()-start,best_epoch=best_epoch,
                  best_validation_score=best,parameter_count=sum(p.numel() for p in net.parameters()),
                  epochs=epochs,training_rows=len(y))
    return bundle,history
