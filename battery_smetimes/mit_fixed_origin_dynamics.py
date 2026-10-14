"""Signal-only frozen MIT origin plus freshly learned relative curve."""
import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from battery_smetimes.monotone_network import CurveNet
from battery_smetimes.mit_segment_features import prefix,predict as feature_predict
from battery_smetimes.tail_selection import predict as old_predict
from battery_smetimes.curve_fidelity import targets

PARAMETERS=dict(block='V',context='segments')


def origin(b,cell):
    w=b['new_weight']
    return float((1-w)*old_predict(b['control'],cell,[101.])[0]+w*feature_predict(b['feature'],cell,[101.])[0])


def fit(cells,ids,width,seed,frozen,epochs=400):
    ids=list(ids);families={cells[k]['physical_family'] for k in ids}
    if len(families)!=1 or frozen['feature']['train_ids']!=ids:raise ValueError('family or origin membership mismatch')
    torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    imputer=SimpleImputer(strategy='median',keep_empty_features=True);scaler=StandardScaler()
    x=torch.tensor(scaler.fit_transform(imputer.fit_transform(np.stack([prefix(cells[k],PARAMETERS) for k in ids]))),dtype=torch.float32)
    indices=[];hs=[];ys=[];ws=[]
    for i,k in enumerate(ids):
        c,y=targets(cells[k]);assert c[0]==101
        ii=np.unique(np.linspace(0,len(c)-1,min(96,len(c))).round().astype(int))
        indices.extend([i]*len(ii));hs.extend((c[ii]-100)/100);ys.extend(y[ii]-y[0])
        w=np.linspace(1,3,len(ii));ws.extend(w/w.sum())
    idx=torch.tensor(indices,dtype=torch.long);h=torch.tensor(hs,dtype=torch.float32)
    y=torch.tensor(ys,dtype=torch.float32);w=torch.tensor(ws,dtype=torch.float32);w/=w.sum()
    net=CurveNet(x.shape[1],width);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.1);history=[]
    for epoch in range(epochs):
        opt.zero_grad();p=net(x,idx,h)-net(x,idx,torch.full_like(h,.01))
        loss=torch.sum(w*(p-y).square())
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        loss.backward();opt.step();history.append(float(loss.detach()))
    return dict(state={k:v.detach().numpy().copy() for k,v in net.state_dict().items()},
        input_width=x.shape[1],width=width,seed=seed,epochs=epochs,history=history,
        imputer=imputer,scaler=scaler,train_ids=ids,family=next(iter(families)),parameters=PARAMETERS,
        frozen=frozen,parameter_count=sum(p.numel() for p in net.parameters()))


def predict(b,cell,future):
    if cell['physical_family']!=b['family']:raise ValueError('different family')
    c=np.asarray(future,dtype=float)
    if c.ndim!=1 or not len(c) or not np.isfinite(c).all() or np.any(c<=100):raise ValueError('invalid future')
    x=b['scaler'].transform(b['imputer'].transform(prefix(cell,b['parameters'])[None]))
    net=CurveNet(b['input_width'],b['width']);net.load_state_dict({k:torch.from_numpy(v) for k,v in b['state'].items()});net.eval()
    with torch.no_grad():
        xx=torch.tensor(x,dtype=torch.float32);idx=torch.zeros(len(c),dtype=torch.long)
        p=net(xx,idx,torch.tensor((c-100)/100,dtype=torch.float32))-net(xx,idx,torch.full((len(c),),.01))
        result=origin(b['frozen'],cell)+p.numpy().astype(float)
    if not np.isfinite(result).all():raise ValueError('invalid forecast')
    return result
