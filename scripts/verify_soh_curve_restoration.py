"""Replay the released interface and check meaningful input rejection cases."""
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from battery_smetimes.physical_types import load_registry
from scripts.diagnose_soh_curve_small import save
from scripts.restore_soh_curve_comparison import load_cells
from scripts.predict_restored_soh import predict


def main():
    root=Path(sys.argv[1]);config=json.loads((root/'config.json').read_text())
    folder=root/'interface_verification'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');folder.mkdir(parents=True,exist_ok=False)
    release=root/'restored_tree_release.json'
    ids=[next(k for k in config['diagnostic_validation'] if k.startswith(b)) for b in ('b1','b2','b3')]
    cells=load_cells(ids);checks=[]
    specs=load_registry()['families']['A123_APR18650M1A_LFP_1p1Ah']['specifications']
    for key in ids:
        cell=cells[key];take=cell['cycle']<=100
        signal=folder/(key+'_signals.npz');np.savez_compressed(signal,cycle=cell['cycle'][take],raw_features=cell['raw'][take])
        meta={**cell['metadata'],'cell_id':key,'specifications':specs}
        metadata=folder/(key+'_metadata.json');save(metadata,meta)
        result=predict(release,signal,metadata,1100);save(folder/(key+'_forecast.json'),result)
        samples=[]
        for seed in config['seeds']:
            with np.load(root/'tree_large'/str(seed)/'predictions.npz') as saved:
                x=saved[key+'__cycle'];p=saved[key+'__prediction'];take=x<=1100
                samples.append(p[take])
        ix=x[x<=1100].astype(int)-101
        np.testing.assert_allclose(np.asarray(result['predicted_soh'])[ix],np.mean(samples,axis=0),rtol=0,atol=1e-12)
        checks.append(dict(check='interface_saved_prediction_replay',cell_id=key,passed=True,atol=1e-12))
    key=ids[0];signal=folder/(key+'_signals.npz');metadata=folder/(key+'_metadata.json')
    good=json.loads(metadata.read_text())
    for case,change in [('incompatible_chemistry',{'specifications':{**specs,'chemistry':'NCM532'}}),
                        ('missing_specifications',{'specifications':{'chemistry':'LFP/graphite'}}),
                        ('wrong_feature_dataset',{'dataset':'NASA'}),
                        ('training_cell_as_unseen',{'cell_id':config['train_large'][0]})]:
        bad={**good,**change};path=folder/(case+'.json');save(path,bad)
        try:predict(release,signal,path,1100)
        except ValueError as error:checks.append(dict(check=case,passed=True,reason=str(error)))
        else:raise AssertionError('invalid input accepted: '+case)
    with np.load(signal) as data:
        forbidden=folder/'forbidden_label_input.npz';np.savez_compressed(forbidden,cycle=data['cycle'],raw_features=data['raw_features'],soh=np.ones(len(data['cycle'])))
    try:predict(release,forbidden,metadata,1100)
    except ValueError as error:checks.append(dict(check='reject_soh_input',passed=True,reason=str(error)))
    else:raise AssertionError('SOH field accepted')
    save(folder/'checks.json',dict(passed=True,checks=checks))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
