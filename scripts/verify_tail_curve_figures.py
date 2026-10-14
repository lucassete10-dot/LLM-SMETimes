"""Full plotted-data coverage and seed-statistic audit, not visual approval."""
import argparse, json
from pathlib import Path
import numpy as np
from scripts.prepare_same_type_forecast import sha, save_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--folder',type=Path,required=True)
    folder=parser.parse_args().folder;manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['complete'] and manifest['all_cells_included']
    run=Path(manifest['run']);cfg=json.loads((run/'config.json').read_text())
    training=json.loads((Path(cfg['training_source'])/'config.json').read_text())
    expected={k for p in training['plans'] for k in p['test']}
    assert len(manifest['figures'])==len(expected)
    assert {r['cell_id'] for r in manifest['figures']}==expected
    checked=[]
    for record in manifest['figures']:
        key=record['cell_id'];file=folder/(key+'.source.npz')
        assert sha(file)==record['source_sha256']
        assert sha(folder/(key+'.png'))==record['png_sha256']
        with np.load(file) as source:
            for label, mapping in (('new',record['new_predictions_sha256']),('old',record['baseline_predictions_sha256'])):
                assert len(mapping)==5;predictions=[]
                for name,digest in mapping.items():
                    assert sha(name)==digest
                    with np.load(name) as a:
                        for field in ('cycle','truth'):np.testing.assert_array_equal(source[field],a[key+'__100__'+field])
                        predictions.append(a[key+'__100__prediction'])
                values=np.stack(predictions)
                np.testing.assert_allclose(source[label+'_mean'],values.mean(axis=0),atol=1e-12,rtol=0)
                np.testing.assert_allclose(source[label+'_seed_sd'],values.std(axis=0,ddof=1),atol=1e-12,rtol=0)
            assert np.all(np.diff(source['cycle'])>0)
            points=len(source['cycle'])
        for result in record['audits'].values():assert result['exit_code']==0
        collision=json.loads((folder/(key+'.collision.json')).read_text())
        assert collision['auditable'] and collision['verdict']=='PASS'
        alignment=json.loads((folder/(key+'.alignment.json')).read_text())
        checked.append(dict(cell_id=key,points=points,source_sha256=sha(file),
            collision_sha256=sha(folder/(key+'.collision.json')),alignment_sha256=sha(folder/(key+'.alignment.json'))))
    save_json(folder/'independent_figure_data_audit.json',dict(passed=True,cells=checked,
        all_curves_covered=True,no_smoothing_or_downsampling=True,seed_mean_sd_recomputed=True,
        source_sha256=sha(__file__),visual_review_complete=False,goal_complete=False))
    print('FIGURE_DATA_PASS cells='+str(len(checked)))


if __name__=='__main__':main()
