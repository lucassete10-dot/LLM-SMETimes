"""Audit train-only representation fits without invoking a forecasting head."""
import json,sys
from pathlib import Path
import numpy as np
from scripts.train_tail_boost import load_cells,NCA
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.train_nasa_robust_inner import detail
from scripts.verify_nasa_robust_inner import close


def main(root):
    cfg=json.loads((root/'config.json').read_text());plan=cfg['plan']
    cells=load_cells(dict(plan,validation=[],test=[]))
    assert set(cells)==set(cfg['loaded_ids'])==set(plan['train']) and not set(cells)&set(plan['validation']+plan['test'])
    assert sha(Path(cfg['parent'])/'config.json')==cfg['parent_config_sha256']
    assert sha(NCA/'index.json')==cfg['nca_index_sha256']
    for f,d in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/Path(f).name)==d
    rows=json.loads((root/'rows.json').read_text());s=json.loads((root/'summary.json').read_text());assert len(rows)==117
    with np.load(root/'reconstructions.npz',allow_pickle=False) as arrays:
        assert len(arrays.files)==117*4
        for row in rows:
            k,m=row['cell_id'],row['mode'];stem=k+'__'+m+'__'
            c,y,p,a=[arrays[stem+f] for f in ('cycle','truth','reconstruction','coefficients')]
            np.testing.assert_array_equal(c,cells[k]['cycle']);np.testing.assert_array_equal(y,cells[k]['truth'])
            np.testing.assert_array_equal(a,row['coefficients']);h=(c-100)/100
            expected=a[0]-a[1]*h
            if m=='quadratic':expected-=a[2]*h*h
            elif m=='knee':expected-=a[2]*np.maximum(h-a[3],0)**2
            else:assert m=='linear'
            np.testing.assert_allclose(p,expected,atol=1e-12,rtol=0);assert np.all(np.diff(p)<=1e-12)
            close({f:row[f] for f in detail(c,y,p)},detail(c,y,p))
            np.testing.assert_allclose(row['mape_percent'],100*np.mean(abs(p-y)/y),atol=1e-10,rtol=0)
            assert row['fitted_with_own_training_future_labels'] and row['not_forecast']
    for m in ('linear','quadratic','knee'):
        for f,v in s['arms'][m].items():np.testing.assert_allclose(v,np.mean([r[f] for r in rows if r['mode']==m]),atol=1e-10,rtol=0)
    knee=[r for r in rows if r['mode']=='knee'];coverage=s['arms']['training_knee_coverage']
    assert coverage['count']==39 and coverage['active_acceleration']==sum(r['coefficients'][2]>1e-6 for r in knee)
    assert coverage['knee_inside_observed_future']==sum(r['coefficients'][2]>1e-6 and 100+100*r['coefficients'][3]<cells[r['cell_id']]['cycle'][-1] for r in knee)
    assert s['predictive_fits']==0 and not s['outer_loaded'] and not s['validation_loaded']
    save_json(root/'independent_verification.json',dict(passed=True,curves=117,train_cells=39,outer_loaded=False,validation_loaded=False,
        scope='Source/data isolation, reconstruction formula, monotonicity and metrics; not an independent optimizer refit or forecast evaluation.',
        verifier_sha256=sha(Path(__file__))))
    print('PASS:117 TRAIN reconstructions, no validation/test labels; not forecast accuracy.')


if __name__=='__main__':main(Path(sys.argv[1]))
