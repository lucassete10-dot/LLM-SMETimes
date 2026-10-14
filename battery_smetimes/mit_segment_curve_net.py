"""Existing small CurveNet, MIT early V segment feature contract."""
import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from battery_smetimes.monotone_network import CurveNet
from battery_smetimes.mit_segment_features import prefix
from battery_smetimes.curve_fidelity import targets

PARAMETERS=dict(block='V',context='segments')


def fit(cells,ids,width,seed,epochs=400):
    ids=list(ids); families={cells[k]['physical_family'] for k in ids}
    if len(families)!=1:raise ValueError('different family')
    torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    imputer=SimpleImputer(strategy='median',keep_empty_features=True)
    scaler=StandardScaler()
    x=torch.tensor(scaler.fit_transform(imputer.fit_transform(np.stack([prefix(cells[k],PARAMETERS) for k in ids]))),dtype=torch.float32)
    indices=[];hs=[];ys=[];ws=[]
    for i,k in enumerate(ids):
        c,y=targets(cells[k]);take=np.unique(np.linspace(0,len(c)-1,min(96,len(c))).round().astype(int))
        indices.extend([i]*len(take));hs.extend((c[take]-100)/100);ys.extend(y[take])
        w=np.linspace(1,3,len(take));ws.extend(w/w.sum())
    index=torch.tensor(indices,dtype=torch.long);h=torch.tensor(hs,dtype=torch.float32)
    y=torch.tensor(ys,dtype=torch.float32);w=torch.tensor(ws,dtype=torch.float32);w/=w.sum()
    net=CurveNet(x.shape[1],width);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.1)
    history=[]
    for epoch in range(epochs):
        opt.zero_grad();p=net(x,index,h);loss=torch.sum(w*(p-y).square())
        if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
        loss.backward();opt.step();history.append(float(loss.detach()))
    return dict(state={k:v.detach().numpy().copy() for k,v in net.state_dict().items()},
        width=width,input_width=x.shape[1],seed=seed,epochs=epochs,history=history,
        imputer=imputer,scaler=scaler,train_ids=ids,family=next(iter(families)),parameters=PARAMETERS,
        parameter_count=sum(p.numel() for p in net.parameters()))


def predict(b,cell,future):
    if cell['physical_family']!=b['family']:raise ValueError('different physical family')
    c=np.asarray(future,dtype=float)
    if c.ndim!=1 or not len(c) or not np.isfinite(c).all() or np.any(c<=100):raise ValueError('invalid future')
    x=b['scaler'].transform(b['imputer'].transform(prefix(cell,b['parameters'])[None]))
    net=CurveNet(b['input_width'],b['width']);net.load_state_dict({k:torch.from_numpy(v) for k,v in b['state'].items()});net.eval()
    with torch.no_grad():
        p=net(torch.tensor(x,dtype=torch.float32),torch.zeros(len(c),dtype=torch.long),torch.tensor((c-100)/100,dtype=torch.float32)).numpy().astype(float)
    if not np.isfinite(p).all():raise ValueError('invalid predictions')
    return p
