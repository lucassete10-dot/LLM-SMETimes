"""Small signal-conditioned smooth monotone degradation curve."""
import numpy as np
import torch
from torch import nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from battery_smetimes.conditional_drift import prefix
from battery_smetimes.tail_boost import labels,predict as tree_predict


class CurveNet(nn.Module):
    def __init__(self,inputs,width):
        super().__init__();self.network=nn.Sequential(nn.Linear(inputs,width),nn.Tanh(),nn.Linear(width,4))
        with torch.no_grad():self.network[-1].bias.copy_(torch.tensor([1.,0.,0.,np.log(np.expm1(.2))]))

    def parameters_for(self,x):
        u=self.network(x)
        return torch.stack((.5+.75*torch.sigmoid(u[:,0]),.02*nn.functional.softplus(u[:,1]),
            .003*nn.functional.softplus(u[:,2]),nn.functional.softplus(u[:,3])),dim=1)

    def forward(self,x,index,h):
        values=self.parameters_for(x)[index];a,b,c,tau=values.unbind(dim=1)
        hinge=.1*nn.functional.softplus((h-tau)/.1)
        return a-b*h-c*hinge.square()


def fit(cells,ids,width,decay,seed,epochs=400):
    if len({cells[k]['physical_family'] for k in ids})!=1:raise ValueError('different physical families')
    torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    raw=np.stack([prefix(cells[k]) for k in ids]);imputer=SimpleImputer(strategy='median',keep_empty_features=True)
    z=imputer.fit_transform(raw);scaler=StandardScaler();z=scaler.fit_transform(z)
    x=torch.tensor(z,dtype=torch.float32);indices=[];hs=[];ys=[];ws=[]
    for i,key in enumerate(ids):
        cycle,y=labels(cells[key]);take=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int))
        indices.extend([i]*len(take));hs.extend((cycle[take]-100)/100);ys.extend(y[take])
        w=np.linspace(1,3,len(take));ws.extend(w/w.sum())
    index=torch.tensor(indices,dtype=torch.long);h=torch.tensor(hs,dtype=torch.float32)
    y=torch.tensor(ys,dtype=torch.float32);w=torch.tensor(ws,dtype=torch.float32);w/=w.sum()
    model=CurveNet(x.shape[1],width);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=decay)
    history=[];model.train()
    for epoch in range(epochs):
        optimizer.zero_grad();p=model(x,index,h);loss=torch.sum(w*(p-y).square())
        if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
        loss.backward();optimizer.step();history.append(float(loss.detach()))
    return dict(mode='network',state={k:v.detach().numpy().copy() for k,v in model.state_dict().items()},
        input_width=x.shape[1],width=width,decay=decay,seed=seed,epochs=epochs,history=history,
        imputer=imputer,scaler=scaler,train_ids=list(ids),family=cells[ids[0]]['physical_family'],
        parameter_count=sum(p.numel() for p in model.parameters()))


def predict(b,cell,cycle):
    if b['mode']=='control':return tree_predict(b['base'],cell,cycle)
    if cell['physical_family']!=b['family']:raise ValueError('different physical family')
    cycle=np.asarray(cycle,dtype=float)
    if cycle.ndim!=1 or not len(cycle) or not np.isfinite(cycle).all() or np.any(cycle<=100):raise ValueError('invalid future cycle request')
    x=b['scaler'].transform(b['imputer'].transform(prefix(cell)[None]))
    model=CurveNet(b['input_width'],b['width']);model.load_state_dict({k:torch.from_numpy(v) for k,v in b['state'].items()});model.eval()
    with torch.no_grad():
        p=model(torch.tensor(x,dtype=torch.float32),torch.zeros(len(cycle),dtype=torch.long),torch.tensor((cycle-100)/100,dtype=torch.float32)).numpy().astype(float)
    if not np.isfinite(p).all():raise ValueError('nonfinite forecast')
    return p
