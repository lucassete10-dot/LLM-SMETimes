"""Training-label-derived initialization; inference remains signal-only."""
import numpy as np
import torch
from scipy.optimize import lsq_linear
from battery_smetimes.monotone_network import fit as base_fit,CurveNet
from battery_smetimes.conditional_drift import prefix
from battery_smetimes.tail_boost import labels


def derive_prior(cells,ids):
    rows=[]
    for key in ids:
        cycle,y=labels(cells[key]);take=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int))
        h=(cycle[take]-100)/100;w=np.linspace(1,3,len(h));w/=w.sum();best=None
        for knee in np.linspace(0,float(h.max()),31):
            hinge=.1*np.logaddexp(0,(h-knee)/.1);design=np.column_stack([np.ones(len(h)),-h,-hinge**2])
            result=lsq_linear(design*np.sqrt(w)[:,None],y[take]*np.sqrt(w),bounds=([.5,0,0],[1.25,np.inf,np.inf]))
            if not result.success:raise ValueError('training parameter fit failed')
            loss=float(np.sum(w*(design@result.x-y[take])**2))
            if best is None or loss<best[0]:best=(loss,knee,result.x)
        loss,knee,values=best
        rows.append(dict(cell_id=key,anchor=float(values[0]),rate=float(values[1]),acceleration=float(values[2]),knee_h=float(knee),sampled_weighted_mse=loss))
    prior=np.median([[r[f] for f in ('anchor','rate','acceleration','knee_h')] for r in rows],axis=0)
    return prior,rows


def inverse_bias(prior):
    a,b,c,tau=prior;fraction=np.clip((a-.5)/.75,1e-6,1-1e-6)
    return np.array([np.log(fraction/(1-fraction)),np.log(np.expm1(max(b,1e-6)/.02)),
        np.log(np.expm1(max(c,1e-6)/.003)),np.log(np.expm1(max(tau,1e-6)))],dtype=np.float32)


def fit(cells,ids,width,decay,seed,epochs):
    b=base_fit(cells,ids,width,decay,seed,epochs=0);prior,rows=derive_prior(cells,ids)
    raw=np.stack([prefix(cells[k]) for k in ids]);z=b['scaler'].transform(b['imputer'].transform(raw));x=torch.tensor(z,dtype=torch.float32)
    indices=[];hs=[];ys=[];ws=[]
    for index,key in enumerate(ids):
        cycle,y=labels(cells[key]);take=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int))
        indices.extend([index]*len(take));hs.extend((cycle[take]-100)/100);ys.extend(y[take]);w=np.linspace(1,3,len(take));ws.extend(w/w.sum())
    index=torch.tensor(indices,dtype=torch.long);h=torch.tensor(hs,dtype=torch.float32);y=torch.tensor(ys,dtype=torch.float32)
    w=torch.tensor(ws,dtype=torch.float32);w/=w.sum()
    model=CurveNet(b['input_width'],width);model.load_state_dict({k:torch.from_numpy(v) for k,v in b['state'].items()})
    bias=inverse_bias(prior)
    with torch.no_grad():model.network[-1].weight.zero_();model.network[-1].bias.copy_(torch.tensor(bias))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=decay);history=[];model.train()
    for epoch in range(epochs):
        optimizer.zero_grad();p=model(x,index,h);loss=torch.sum(w*(p-y).square())
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        loss.backward();optimizer.step();history.append(float(loss.detach()))
    b.update(state={k:v.detach().numpy().copy() for k,v in model.state_dict().items()},epochs=epochs,history=history,
        initialization_prior=prior.tolist(),initialization_bias=bias.tolist(),training_parameter_fits=rows,warm_start=True)
    return b
