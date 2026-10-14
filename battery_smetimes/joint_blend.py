"""One global convex weight, hard family and training-membership isolation."""
import numpy as np
from battery_smetimes.pls_joint import predict as joint_predict
from battery_smetimes.tail_selection import predict as primary_predict
from battery_smetimes.research_baselines import predict_baseline


def make(primary,joint,weight):
    if not np.isfinite(weight) or not 0<=weight<=1:raise ValueError('invalid blend weight')
    if primary['family']!=joint['family'] or primary['train_ids']!=joint['train_ids']:raise ValueError('family or training membership mismatch')
    return dict(primary=primary,joint=joint,weight=float(weight),family=joint['family'],train_ids=joint['train_ids'])


def components(b,cell,cycle):
    if cell['physical_family']!=b['family']:raise ValueError('different physical family')
    old=b['primary']
    p=predict_baseline(old['inner'],cell,100,cycle) if old['kind']=='research_baseline' else primary_predict(old,cell,cycle)
    q=joint_predict(b['joint'],cell,cycle)
    return p,q


def predict(b,cell,cycle):
    p,q=components(b,cell,cycle);y=(1-b['weight'])*p+b['weight']*q
    if not np.isfinite(y).all():raise ValueError('nonfinite blend prediction')
    return y
