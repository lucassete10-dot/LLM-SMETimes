"""Known-protocol local models, with explicit compatible-family fallback."""
import json
from collections import defaultdict
from battery_smetimes.curve_fidelity import condition_key
from battery_smetimes import continuous_length as point
from battery_smetimes import integrated_length as rate


def key(cell):
    if 'metadata' not in cell:raise ValueError('known protocol metadata required')
    c=condition_key(cell)
    if not c[0]:raise ValueError('missing known protocol')
    return json.dumps(c)


def fit(cells,ids,targets,origin,kind,seed,minimum=3,local=True):
    ids=list(ids);anchor=point.fit(cells,ids,origin,'point','VIT',1,seed,targets)
    base=anchor if kind=='point' else rate.fit(cells,ids,targets,origin,'VIT',25,1,seed,anchor)
    groups=defaultdict(list)
    for k in ids:groups[key(cells[k])].append(k)
    children={c:fit(cells,keys,targets,origin,kind,seed,minimum,False) for c,keys in groups.items() if local and len(keys)>=minimum}
    return dict(kind=kind,base=base,local=children,train_ids=ids,family=base['family'],
                origin=origin,seed=seed,minimum=minimum,training_group_counts={k:len(v) for k,v in groups.items()})


def route(b,cell):
    if cell['physical_family']!=b['family']:raise ValueError('physical family mismatch')
    c=key(cell)
    return b['local'].get(c,b),c in b['local']


def predict(b,cell,cycle):
    child,_=route(b,cell);fn=point.predict if b['kind']=='point' else rate.predict
    return fn(child['base'],cell,cycle)
