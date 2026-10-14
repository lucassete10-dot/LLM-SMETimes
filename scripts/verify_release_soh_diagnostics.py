"""Verify full diagnostic coverage and label-free frozen-weight predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
from battery_smetimes.external_soh import load_context, load_bundle
from battery_smetimes.research_baselines import regression_metrics
from scripts.train_stm32_compact_trees import predict
from scripts.prepare_same_type_forecast import save_json, sha


def verify(folder):
    config = json.loads((folder/'config.json').read_text())
    progress = json.loads((folder/'progress.json').read_text())
    if not progress['complete']:
        raise ValueError('diagnostic export incomplete')
    if config['independent_test'] is not False or config['role']!='development_training_fit_only':
        raise ValueError('training diagnostics mislabelled as independent tests')
    package = Path(config['package'])
    if sha(package/'manifest.json')!=config['release_manifest_sha256']:
        raise ValueError('release manifest changed')
    context = load_context(package)
    checked, points = 0, 0
    for domain in config['datasets']:
        bundle = load_bundle(context, domain)
        report = json.loads((folder/domain/'metrics.json').read_text())
        expected_ids = set(bundle['train_ids'])
        scored = {r['cell_id'] for r in report['by_cell']}
        skipped = {r['cell_id'] for r in report['skipped']}
        if scored&skipped or scored|skipped!=expected_ids:
            raise ValueError('missing, duplicate, or unaccounted development cells')
        if len(scored)!=len(report['by_cell']):
            raise ValueError('duplicate scored cell')
        with np.load(folder/domain/'predictions.npz', allow_pickle=False) as arrays:
            expected_keys = {f'{k}__100__{f}' for k in scored for f in ('cycle','truth','prediction')}
            if set(arrays.files)!=expected_keys:
                raise ValueError('saved arrays do not match scored membership')
            for row in report['by_cell']:
                key = row['cell_id'];cell = context['cells'][key];prefix=f'{key}__100__'
                x,y,p = [arrays[prefix+f] for f in ('cycle','truth','prediction')]
                mask=np.isfinite(cell['soh'])&(cell['soh']>0)&(cell['label_cycle']>100)
                np.testing.assert_array_equal(x,cell['label_cycle'][mask])
                np.testing.assert_array_equal(y,cell['soh'][mask])
                signal_only={k:v for k,v in cell.items() if k not in ('label_cycle','soh','unfiltered_soh')}
                np.testing.assert_allclose(p,predict(bundle,signal_only,100,x),rtol=0,atol=1e-12)
                for name,value in regression_metrics(y,p).items():
                    if value is None:
                        assert row['metrics'][name] is None
                    else:
                        np.testing.assert_allclose(value,row['metrics'][name],rtol=0,atol=1e-12)
                for extension in ('png','svg','pdf','alignment.json','text-run.json','collision-run.json'):
                    if not (folder/domain/'curves'/f'{key}.{extension}').is_file():
                        raise ValueError('missing cell curve or QA')
                for name in ('text','collision'):
                    result=json.loads((folder/domain/'curves'/f'{key}.{name}-run.json').read_text())
                    if result['exit_code']!=0:raise ValueError('figure automated QA failed')
                checked+=1;points+=len(x)
    if checked!=progress['exported']:raise ValueError('exported coverage count inconsistent')
    return dict(passed=True,cells=checked,targets=points,release_manifest_unchanged=True,
                all_valid_future_labels_preserved=True,label_free_prediction_replay=True,
                figures_automated_qa_checked=True,visual_review='not proven by this verifier',
                role='development_training_fit_only',independent_test=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--diagnostics',type=Path,required=True)
    args=parser.parse_args()
    result=verify(args.diagnostics)
    save_json(args.diagnostics/'verification.json',result)
    print(json.dumps(result,ensure_ascii=False))
