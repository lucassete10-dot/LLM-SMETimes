"""A scalar, train-OOF shrinkage correction preserving predicted cycle101."""
import numpy as np
from battery_smetimes.xjtu_segment_features import predict as base_predict


def gain(records,penalty):
    xx=xy=0.
    for r in records:
        y=np.asarray(r['truth']);p=np.asarray(r['prediction'])
        take=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int))
        w=np.linspace(1,3,len(take));w/=w.sum()*len(records)
        dp=p[0]-p[take];dy=y[0]-y[take]
        xx+=float(np.dot(w,dp*dp));xy+=float(np.dot(w,dp*dy))
    value=float(np.clip((xy+penalty)/(xx+penalty),.5,2))
    return value,dict(sum_weighted_pred_squared=xx,sum_weighted_pred_true=xy,penalty=penalty)


def predict(b,cell,cycles):
    if cell['physical_family']!=b['base']['family']:raise ValueError('different family')
    p=base_predict(b['base'],cell,cycles);a=base_predict(b['base'],cell,[101.])[0]
    return a+b['gain']*(p-a)
