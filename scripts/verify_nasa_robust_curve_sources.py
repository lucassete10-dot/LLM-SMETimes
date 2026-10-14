"""Replay displayed seed means exactly, rather than comparing one run to mean."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.prepare_same_type_forecast import sha,save_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--figures',type=Path,required=True);root=ap.parse_args().figures;m=json.loads((root/'manifest.json').read_text());assert m['complete'] and m['all_cells_included'];assert {r['cell_id'] for r in m['figures']}=={'B0005','B0006','B0007','B0018'};rows=[]
    for r in m['figures']:
        key=r['cell_id'];file=Path(r['figure']+'.source.npz');assert sha(file)==r['source_npz_sha256']
        with np.load(file) as source:
            for prefix,field in [('new','new_prediction_sha256'),('old','old_prediction_sha256')]:
                predictions=[]
                for name,digest in r[field].items():
                    assert sha(name)==digest
                    with np.load(name) as a:
                        np.testing.assert_array_equal(a[key+'__100__cycle'],source['cycle']);np.testing.assert_array_equal(a[key+'__100__truth'],source['truth']);predictions.append(a[key+'__100__prediction'])
                assert len(predictions)==5;stack=np.stack(predictions);np.testing.assert_array_equal(stack.mean(axis=0),source[prefix+'_mean']);np.testing.assert_array_equal(stack.std(axis=0,ddof=1),source[prefix+'_seed_sd']);assert all(np.array_equal(p,predictions[0]) for p in predictions)
            assert r['short_coverage']==bool(len(source['cycle'])<10 or source['cycle'][-1]-source['cycle'][0]<100)
        assert all(r['audits'][k]['exit_code']==0 for k in ('text','collision'));alignment=json.loads(Path(r['figure']+'.alignment.json').read_text());rows.append(dict(cell_id=key,source_sha256=sha(file),short_coverage=r['short_coverage'],geometry_passed=True,alignment=alignment.get('verdict',alignment.get('status'))))
    save_json(root/'independent_source_audit.json',dict(passed=True,rows=rows,source_sha256=sha(__file__),scope='All four saved displayed source curves and exact five-run mean/SD; rendered geometry from current manifest. Visual review separate.',goal_complete=False));print('FOUR_FIGURE_SOURCES_PASS')


if __name__=='__main__':main()
