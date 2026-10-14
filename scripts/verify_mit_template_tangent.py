"""Separate tangent boundary/derivative and all saved-change arithmetic."""
import json,sys
from pathlib import Path
import joblib,numpy as np
from scripts.verify_mit_template_route_loo import errors
from scripts.verify_nasa_robust_inner import close
from scripts.prepare_same_type_forecast import sha,save_json


def relative(t,c):
    _,r,b,k=t['parameters'];end=t['record_end'];h=(c-100)/100;he=(end-100)/100
    p=-r*(h-.01)-b*(np.maximum(h-k,0)**2-max(.01-k,0)**2)
    beyond=c>end
    boundary=-r*(he-.01)-b*(max(he-k,0)**2-max(.01-k,0)**2)
    p[beyond]=boundary+(-r-2*b*max(he-k,0))*(c[beyond]-end)/100
    return p


def main(root):
    cfg=json.loads((root/'config.json').read_text());parent=Path(cfg['parent']);plan=cfg['plan']
    for f,d in cfg['input_sha256'].items():assert sha(Path(f))==d
    for f,d in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/Path(f).name)==d
    pc=json.loads((parent/'config.json').read_text());assert pc['plan']==plan
    assert json.loads((parent/'independent_verification.json').read_text())['passed']
    cache=joblib.load(pc['cache_file']);templates={t['cell_id']:t for t in cache['templates']}
    routes=json.loads((parent/'rows.json').read_text());records=json.loads((root/'rows.json').read_text());ids=plan['train']
    assert len(routes)==len(records)==len(ids)==83;rows=[]
    with np.load(parent/'curves.npz',allow_pickle=False) as old,np.load(root/'curves.npz',allow_pickle=False) as out:
        assert set(out.files)=={k+'__'+f for k in ids for f in ('cycle','truth_change','baseline_change','tangent_change')}
        for pos,k in enumerate(ids):
            rr=routes[pos];assert rr['cell_id']==k and k not in rr['neighbor_ids'] and set(rr['neighbor_ids'])<=set(ids)
            c,y,p,b=[out[k+'__'+f] for f in ('cycle','truth_change','tangent_change','baseline_change')]
            np.testing.assert_array_equal(c,old[k+'__cycle']);np.testing.assert_array_equal(y,old[k+'__truth_change']);np.testing.assert_array_equal(b,old[k+'__nearest_change'])
            expected=sum(w*relative(templates[d],c) for d,w in zip(rr['neighbor_ids'],rr['weights']))
            np.testing.assert_allclose(p,expected,rtol=0,atol=1e-12);np.testing.assert_allclose(p[0],0,rtol=0,atol=1e-12)
            assert np.all(np.diff(p)<=1e-10)
            r=dict(cell_id=k,baseline=errors(b,y),tangent=errors(p,y),majority_neighbors_extrapolated=rr['majority_neighbors_extrapolated'])
            assert records[pos]['cell_id']==k and records[pos]['majority_neighbors_extrapolated']==r['majority_neighbors_extrapolated']
            close(records[pos]['baseline'],r['baseline']);close(records[pos]['tangent'],r['tangent']);rows.append(r)
    summary=json.loads((root/'summary.json').read_text());fields=tuple(rows[0]['baseline'])
    for arm in ('baseline','tangent'):close(summary[arm],{f:float(np.mean([r[arm][f] for r in rows])) for f in fields})
    assert summary['endpoint_improved_cells']==sum(r['tangent']['change_endpoint_pp']<r['baseline']['change_endpoint_pp'] for r in rows)
    assert summary['cells']==83 and summary['curves']==166 and summary['new_fits']==0 and not summary['adopted']
    assert not cfg['validation_loaded'] and not cfg['outer_loaded']
    save_json(root/'independent_verification.json',dict(passed=True,cells=83,curves=166,new_fits=0,adopted=False,outer_loaded=False,
        scope='Separate relative boundary/analytic derivative extrapolation and all change metrics; frozen TRAIN LOO routes from verified Stage88.',verifier_sha256=sha(Path(__file__))))
    print('PASS:83TRAIN held cells,166changes, locked routes, derivative extension and metrics/SHA')


if __name__=='__main__':main(Path(sys.argv[1]))
