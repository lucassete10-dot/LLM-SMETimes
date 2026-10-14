"""Independent metric/coverage verification of paired MIT development probe."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from scripts.train_continuous_length_pilot import evaluate
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);root=ap.parse_args().run
    cfg=json.loads((root/'config.json').read_text());rows=json.loads((root/'all_rows.json').read_text())
    assert json.loads((root/'progress.json').read_text())['success']
    snapshots=root/'source_snapshot';snapshots.mkdir(exist_ok=True)
    for name,digest in cfg['source_sha256'].items():
        assert sha(Path(name))==digest;shutil.copy2(name,snapshots/Path(name).name)
    count=0;groups=[]
    dev=set(cfg['plans'][0]['train']+cfg['plans'][0]['validation'])
    for n in cfg['origins']:
        for seed in cfg['seeds']:
            for arm in cfg['arms']:
                covered=[]
                for plan in cfg['plans']:
                    folder=root/plan['name']/str(n)/str(seed)/arm
                    run=json.loads((folder/'config.json').read_text())
                    assert not set(run['train'])&set(run['validation'])
                    assert not set(run['train']+run['validation'])&set(plan['test'])
                    assert run['future_poison_and_reload_passed']
                    assert sha(folder/'model.joblib')==run['model_sha256']
                    with np.load(folder/'predictions.npz') as d:
                        for cid in run['validation']:
                            c,y,p=[d[cid+'__'+k] for k in ['cycle','truth','prediction']]
                            assert np.all(c>n) and np.isfinite(p).all()
                            mask=c>100;fresh=evaluate(c[mask],y[mask],p[mask])
                            row=next(r for r in rows if (r['cell_id'],r['origin'],r['seed'],r['arm'])==(cid,n,seed,arm))
                            for k in ['mape_percent','mae_soh','rmse_soh','late20_mae_pp','stage_error_pp','endpoint_pp']:
                                np.testing.assert_allclose(fresh[k],row[k],atol=1e-10,rtol=0)
                            covered.append(cid);count+=1
                assert len(covered)==len(set(covered)) and set(covered)==dev
                rr=[r for r in rows if (r['origin'],r['seed'],r['arm'])==(n,seed,arm)]
                groups.append(dict(origin=n,seed=seed,arm=arm,cells=len(rr),
                    **{k:float(np.mean([r[k] for r in rr])) for k in ['mape_percent','late20_mae_pp','stage_error_pp']},
                    mape_pass=sum(r['mape_percent']<=6 for r in rr),strict_screen_pass=sum(r['screen_pass'] for r in rr)))
    save_json(root/'verification.json',dict(passed=True,checked_predictions=count,independent_test=False))
    save_json(root/'summary.json',groups)
    print(json.dumps(groups,indent=2))


if __name__=='__main__':main()
