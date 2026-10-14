"""Inspect trusted local archived bundles for membership; no tree inference."""
import json,zipfile
from pathlib import Path
import joblib
from scripts.audit_llm_curve_repair import fingerprint
from scripts.prepare_same_type_forecast import save_json


def main():
    previous=Path('experiments/uconn_sparse_probe/20261008T073624_949393Z')
    config=json.loads((previous/'config.json').read_text());reserved=set(config['test']);training=set(config['train'])
    assert len(reserved)==11 and not reserved&set(config['train']+config['validation'])
    rows=[];prediction_events=[]
    for path in sorted(previous.glob('*/*/model.joblib')):
        # These are locally generated archived artifacts. Inspect membership
        # only: do not predict, refit, extract a teacher, or load target cells.
        bundle=joblib.load(path);used=set(bundle['train_ids'])
        assert not used&reserved and used==training
        rows.append({'model':str(path),'sha256':fingerprint(path),'train_ids':sorted(used),
            'reserved_train_overlap':sorted(used&reserved)})
        del bundle
    for path in sorted(previous.glob('*/*/predictions.npz')):
        with zipfile.ZipFile(path) as z:
            ids={n.split('__')[0] for n in z.namelist() if n.endswith('__prediction.npy')}
        assert not ids&reserved
        prediction_events.append({'archive':str(path),'predicted_ids':sorted(ids),'reserved_overlap':sorted(ids&reserved)})
    assert len(rows)==len(prediction_events)==16
    p0=Path(json.loads(Path('experiments/llm_curve_repair/latest_P0.json').read_text())['folder'])
    report={'scope':'Historical UConn local artifact membership audit only; no target labels or inference',
        'reserved_ids':sorted(reserved),'archived_models_inspected':rows,'prediction_headers_inspected':prediction_events,
        'previous_source_sha256':fingerprint(previous/'source_snapshot/train_uconn_sparse_probe.py'),
        'previous_config_sha256':fingerprint(previous/'config.json'),'source_sha256':fingerprint(Path(__file__)),
        'reserved_excluded_from_these_saved_fits_and_predictions':True,
        'independent_confirmation_proven':False,
        'limitations':['This proves exclusion from this archived run, not all external or prior viewing.',
            'The previous config explicitly says independent_test=false.',
            'Reference capacity and physical protocol mapping still require certification; processed I/T unavailable.',
            'No archived tree predictor is used in the LLM forecast or as a teacher.'],
        'goal_complete':False}
    save_json(p0/'UConn_reserved_membership_audit.json',report)
    print(json.dumps({'reserved_cells':len(reserved),'saved_models_audited':len(rows),
        'prediction_headers_audited':len(prediction_events),'independent_confirmation_proven':False},indent=2))


if __name__=='__main__':main()
