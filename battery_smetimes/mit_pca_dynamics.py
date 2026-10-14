"""Train-only PCA, affine rate/knee head and immutable signal origin."""
import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from battery_smetimes.mit_segment_features import prefix
from battery_smetimes.mit_fixed_origin_dynamics import origin,PARAMETERS
from battery_smetimes.curve_fidelity import targets


def relative_curve(head,z,index,h):
    u=head(z)[index]
    r=.02*torch.nn.functional.softplus(u[:,0])
    a=.003*torch.nn.functional.softplus(u[:,1])
    t=torch.nn.functional.softplus(u[:,2])
    f=.1*torch.nn.functional.softplus((h-t)/.1)
    f0=.1*torch.nn.functional.softplus((.01-t)/.1)
    return -r*(h-.01)-a*(f.square()-f0.square())


def fit(cells,ids,components,seed,frozen,epochs=1600):
    ids=list(ids);families={cells[k]['physical_family'] for k in ids}
    if len(families)!=1 or frozen['feature']['train_ids']!=ids:raise ValueError('origin membership/family')
    torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    imputer=SimpleImputer(strategy='median',keep_empty_features=True);scaler=StandardScaler()
    raw=np.stack([prefix(cells[k],PARAMETERS) for k in ids]);std=scaler.fit_transform(imputer.fit_transform(raw))
    pca=PCA(n_components=components,svd_solver='full');z=torch.tensor(pca.fit_transform(std),dtype=torch.float32)
    indices=[];hs=[];ys=[];ws=[]
    for i,k in enumerate(ids):
        c,y=targets(cells[k]);assert c[0]==101
        take=np.unique(np.linspace(0,len(c)-1,min(96,len(c))).round().astype(int))
        indices.extend([i]*len(take));hs.extend((c[take]-100)/100);ys.extend(y[take]-y[0])
        w=np.linspace(1,3,len(take));ws.extend(w/w.sum())
    idx=torch.tensor(indices,dtype=torch.long);h=torch.tensor(hs,dtype=torch.float32)
    y=torch.tensor(ys,dtype=torch.float32);w=torch.tensor(ws,dtype=torch.float32);w/=w.sum()
    head=torch.nn.Linear(components,3)
    with torch.no_grad():head.bias.copy_(torch.tensor([0.,0.,np.log(np.expm1(.2))]))
    opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.1);history=[]
    for epoch in range(epochs):
        opt.zero_grad();p=relative_curve(head,z,idx,h);loss=torch.sum(w*(p-y).square())
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        loss.backward();opt.step();history.append(float(loss.detach()))
    return dict(state={k:v.detach().numpy().copy() for k,v in head.state_dict().items()},components=components,
        seed=seed,epochs=epochs,history=history,imputer=imputer,scaler=scaler,pca=pca,train_ids=ids,
        family=next(iter(families)),frozen=frozen,parameters=PARAMETERS,head_parameter_count=sum(p.numel() for p in head.parameters()))


def predict(b,cell,future):
    if cell['physical_family']!=b['family']:raise ValueError('different family')
    c=np.asarray(future,dtype=float)
    if c.ndim!=1 or not len(c) or not np.isfinite(c).all() or np.any(c<=100):raise ValueError('invalid future')
    z=b['pca'].transform(b['scaler'].transform(b['imputer'].transform(prefix(cell,b['parameters'])[None])))
    head=torch.nn.Linear(b['components'],3);head.load_state_dict({k:torch.from_numpy(v) for k,v in b['state'].items()})
    with torch.no_grad():
        change=relative_curve(head,torch.tensor(z,dtype=torch.float32),torch.zeros(len(c),dtype=torch.long),
            torch.tensor((c-100)/100,dtype=torch.float32)).numpy().astype(float)
    p=origin(b['frozen'],cell)+change
    if not np.isfinite(p).all():raise ValueError('invalid prediction')
    return p
