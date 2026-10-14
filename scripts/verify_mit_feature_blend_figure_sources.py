"""All MIT figure source stacks and rendered geometry, no model fitting."""
import argparse,json
from pathlib import Path
import fitz,numpy as np
from scripts.prepare_same_type_forecast import sha,save_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--figures',type=Path,required=True);ap.add_argument('--run',type=Path,required=True)
    args=ap.parse_args();root=args.run;folder=args.figures
    cfg=json.loads((root/'config.json').read_text());manifest=json.loads((folder/'manifest.json').read_text())
    old=Path(cfg['current_primary']);assert manifest['outer_verification_sha256']==sha(root/'independent_verification.json')
    assert manifest['complete'] and manifest['all_cells_included']
    expected={k for p in cfg['plans'] for k in p['test']}
    assert len(expected)==134 and expected=={r['cell_id'] for r in manifest['figures']} and len(manifest['figures'])==134
    bycell={r['cell_id']:r for r in json.loads((root/'by_cell.json').read_text())};reports=[]
    for r in manifest['figures']:
        key=r['cell_id'];npz=folder/(key+'.source.npz');assert sha(npz)==r['source_npz_sha256'] and r['metrics']==bycell[key]
        stacks=[];xs=[];ys=[]
        for source,field in [(root,'new_prediction_sha256'),(old,'old_prediction_sha256')]:
            preds=[]
            for seed in cfg['seeds']:
                file=source/r['fold']/str(seed)/'test_predictions.npz';assert sha(file)==r[field][str(file.resolve())]
                with np.load(file) as a:
                    x,y,p=[a[key+'__100__'+f] for f in ('cycle','truth','prediction')]
                    xs.append(x);ys.append(y);preds.append(p)
            stacks.append(np.stack(preds))
        assert all(np.array_equal(xs[0],x) for x in xs) and all(np.array_equal(ys[0],y) for y in ys)
        with np.load(npz) as a:
            np.testing.assert_array_equal(a['cycle'],xs[0]);np.testing.assert_array_equal(a['truth'],ys[0])
            for name,stack in zip(('new','old'),stacks):
                np.testing.assert_array_equal(a[name+'_seed_predictions'],stack)
                np.testing.assert_array_equal(a[name+'_mean'],stack.mean(axis=0))
                np.testing.assert_array_equal(a[name+'_seed_sd'],stack.std(axis=0,ddof=1))
        for name in ('text','collision'):assert r['audits'][name]['exit_code']==0
        collision=json.loads((folder/(key+'.collision.json')).read_text())
        assert collision['auditable'] and collision['summary']['fail']==collision['summary']['warn']==0
        text=json.loads(json.loads((folder/(key+'.text-run.json')).read_text())['stdout'])
        assert text['minimum_found_pt']>=8 and text['below_minimum_count']==0
        with fitz.open(folder/(key+'.pdf')) as pdf:
            assert pdf.page_count==1
            width=pdf[0].rect.width/72*25.4;height=pdf[0].rect.height/72*25.4
            assert abs(width-183)<1e-4 and abs(height-110)<1e-4
        assert '<text' in (folder/(key+'.svg')).read_text(encoding='utf-8')
        reports.append(dict(cell_id=key,source_verified=True,pdf_width_mm=width,pdf_height_mm=height,
            minimum_font_pt=text['minimum_found_pt'],collision_passed=True))
    with fitz.open(folder/'all_cells.pdf') as pdf:assert pdf.page_count==134
    for name in ('text','collision'):assert manifest['combined_pdf_audits'][name]['exit_code']==0
    save_json(folder/'independent_source_verification.json',dict(passed=True,cells=134,
        seed_prediction_arrays=1340,combined_pdf_pages=134,panels=reports,source_sha256=sha(__file__),
        scope='all arrays, variability, rendered geometry and automated audits; human visual review separate',goal_complete=False))
    print('PASSED 134 figure source/geometry/automated QA checks',flush=True)


if __name__=='__main__':main()
