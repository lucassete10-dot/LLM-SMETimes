"""Frozen supervised signal representation, joint functional regression loss."""
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
from battery_smetimes.conditional_drift import prefix
from battery_smetimes.parameter_ridge import trajectory
from battery_smetimes.tail_boost import labels,predict as tree_predict


def scores(encoder,cell):
    if cell['physical_family']!=encoder['family']:raise ValueError('different physical family')
    x=encoder['scaler'].transform(encoder['imputer'].transform(prefix(cell)[None]))
    return encoder['model'].transform(x)[0]


def fit(cells,ids,encoder,decay,seed,epochs=2000):
    ids=list(ids)
    if encoder['train_ids']!=ids or not encoder['logarithmic']:raise ValueError('initializer membership/transform mismatch')
    torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    latent=np.stack([scores(encoder,cells[k]) for k in ids]);scaler=StandardScaler();z=scaler.fit_transform(latent)
    raw=np.stack([prefix(cells[k]) for k in ids]);x=encoder['scaler'].transform(encoder['imputer'].transform(raw))
    u=encoder['model'].predict(x)*encoder['target_scale']+encoder['target_mean']
    design=np.column_stack([z,np.ones(len(z))]);initial=np.linalg.lstsq(design,u,rcond=None)[0]
    np.testing.assert_allclose(design@initial,u,atol=1e-9,rtol=1e-9)
    weight=torch.nn.Parameter(torch.tensor(initial[:-1].T,dtype=torch.float64));bias=torch.nn.Parameter(torch.tensor(initial[-1],dtype=torch.float64))
    zz=torch.tensor(z,dtype=torch.float64);indices=[];hs=[];ys=[];ws=[]
    for i,key in enumerate(ids):
        cycle,y=labels(cells[key]);idx=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int))
        indices.extend([i]*len(idx));hs.extend((cycle[idx]-100)/100);ys.extend(y[idx]);w=np.linspace(1,3,len(idx));ws.extend(w/w.sum())
    index=torch.tensor(indices,dtype=torch.long);h=torch.tensor(hs,dtype=torch.float64);y=torch.tensor(ys,dtype=torch.float64)
    w=torch.tensor(ws,dtype=torch.float64);w/=w.sum()
    optimizer=torch.optim.AdamW([dict(params=[weight],weight_decay=decay),dict(params=[bias],weight_decay=0.)],lr=.001)
    history=[]
    for epoch in range(epochs):
        optimizer.zero_grad();uu=(zz@weight.T+bias)[index];a=.5+.75*torch.sigmoid(uu[:,0]);rates=torch.exp(uu[:,1:])
        hinge=.1*torch.nn.functional.softplus((h-rates[:,2])/.1);p=a-rates[:,0]*h-rates[:,1]*hinge.square()
        loss=torch.sum(w*(p-y).square())
        if not torch.isfinite(loss):raise ValueError('nonfinite joint loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_([weight,bias],5.)
        if not torch.isfinite(norm):raise ValueError('nonfinite joint gradient')
        optimizer.step();history.append(float(loss.detach()))
    return dict(mode='joint',encoder=encoder,latent_scaler=scaler,coefficient=weight.detach().numpy().copy(),intercept=bias.detach().numpy().copy(),initialization=initial,history=history,epochs=epochs,decay=decay,seed=seed,train_ids=ids,family=encoder['family'],head_parameter_count=initial.size)


def predicted_parameters(b,cell):
    z=b['latent_scaler'].transform(scores(b['encoder'],cell)[None])[0];u=z@b['coefficient'].T+b['intercept']
    with np.errstate(over='raise',invalid='raise'):p=np.r_[.5+.75*expit(u[0]),np.exp(u[1:])]
    if not np.isfinite(p).all():raise ValueError('nonfinite parameters')
    return p


def predict(b,cell,cycle):
    if b['mode']=='control':return tree_predict(b['base'],cell,cycle)
    return trajectory(predicted_parameters(b,cell),cycle)
