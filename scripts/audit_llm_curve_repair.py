"""Freeze P0 provenance, TRAIN diagnostic selection and raw b2c1 semantics."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import numpy as np

from battery_smetimes.llm_curve_objective import training_origin_target
from battery_smetimes.llm_knee_supervision import training_knee_target
from scripts.prepare_same_type_forecast import save_json, mit_prefix
from scripts.train_same_type_forecast import load_data
from prepare_mit import read_ref
from battery_smetimes.mit_units import mit_prefix_to_amperes,MIT_NOMINAL_AMPERES_PER_C


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_frozen(settings):
    if 'unified_external_index' in settings:
        from scripts.prepare_llm_external_datasets import load_prepared_external
        return load_prepared_external(settings)
    audit = Path(settings['frozen_plan'])
    frozen = json.loads((audit/'config.json').read_text(encoding='utf-8'))
    prepared = Path(json.loads((audit/'audit.json').read_text(encoding='utf-8'))['prepared_folder'])
    registry_path = audit/'source_snapshot/battery_physical_registry.json'
    checks = [(registry_path, frozen['registry_sha256']),
              (prepared/'manifest.json', frozen['prepared_manifest_sha256']),
              (Path(frozen['config']['integrity_audit']), frozen['integrity_sha256'])]
    for file, expected in checks:
        if fingerprint(file) != expected:
            raise ValueError(f'frozen source drift: {file}')
    manifest, cells = load_data(prepared, frozen['config']['integrity_audit'])
    registered = json.loads((audit/'cell_registry.json').read_text(encoding='utf-8'))['cells']
    for key, cell in cells.items():
        cell.update({field: registered[key][field] for field in
                     ('physical_family', 'reference_established_source_cycle')})
        if settings.get('mit_current_units')=='physical_amperes_v2' and cell['metadata']['dataset']=='MIT':
            cell['raw']=mit_prefix_to_amperes(cell['raw'])
            cell['current_scale_to_ampere']=MIT_NOMINAL_AMPERES_PER_C
        elif settings.get('mit_current_units','legacy_stored_v1') not in ('legacy_stored_v1','physical_amperes_v2'):
            raise ValueError('unknown MIT unit contract')
    splits = json.loads((audit/'splits.json').read_text(encoding='utf-8'))['plans']
    plan = next(p for p in splits if p['name'] == settings['fold'])
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        if set(plan[a]) & set(plan[b]):
            raise ValueError('physical cell split overlap')
    family = cells[plan['train'][0]]['physical_family']
    if any(cells[k]['physical_family'] != family for role in ('train', 'validation', 'test') for k in plan[role]):
        raise ValueError('mixed incompatible physical families')
    return manifest, cells, plan, registry_path, prepared


def training_inventory(cells, ids, origin):
    result = []
    for cid in sorted(ids):
        c = cells[cid]; visible = c['cycle'] <= origin
        valid = (c['label_cycle'] > origin) & np.isfinite(c['soh']) & (c['soh'] > 0)
        x, y = c['label_cycle'][valid], c['soh'][valid]
        start, available = training_origin_target(c, origin)
        if len(y) < 9:
            result.append({'cell_id': cid, 'eligible': False, 'reason': 'insufficient TRAIN target coverage'})
            continue
        span = float(x[-1]-x[0]); width = min(5., span/10)
        initial, final = np.median(y[x<=x[0]+width]), np.median(y[x>=x[-1]-width])
        knee = training_knee_target(c, origin)
        rate = 100*float(initial-final)/span
        early = float(np.median(y[x<=x[0]+.2*span]))
        late = float(np.median(y[x>=x[0]+.8*span]))
        category = ('recovery' if late > early+.005 else 'accelerating' if knee['valid']
                    else 'fast' if rate >= .025 else 'slow')
        raw = c['raw'][visible]
        result.append({'cell_id':cid, 'eligible':bool(available and len(raw)>=3),
            'category':category, 'classification_scope':'TRAIN target-only diagnostic selection, never inference routing',
            'visible_observations':len(raw), 'visible_first_cycle':int(c['cycle'][visible][0]),
            'visible_last_cycle':int(c['cycle'][visible][-1]), 'missing_feature_fraction':float(np.mean(~np.isfinite(raw))),
            'source_cycle_gap_max':int(np.max(np.diff(c['cycle'][visible]))) if len(raw)>1 else None,
            'exact_origin_soh':start if available else None, 'origin_label_available':bool(available),
            'future_points':len(x), 'future_span':span, 'drop_pp':100*float(initial-final),
            'drop_soh_per_100cycles':rate, 'knee':knee,
            'conditions':c['metadata'], 'reference_established_source_cycle':c['reference_established_source_cycle']})
    return result


def select_tiny(rows, count):
    if not 8 <= count <= 16:
        raise ValueError('tiny diagnosis must cover 8 to 16 TRAIN cells')
    eligible = [r for r in rows if r['eligible']]
    selected=[]
    # Deterministic category round-robin, sampling quantiles of observed TRAIN
    # rate; includes different curve regimes without consulting heldout labels.
    buckets = {}
    for cat in ('fast','accelerating','slow','recovery'):
        ordered=sorted([r for r in eligible if r['category']==cat], key=lambda r:(r['drop_soh_per_100cycles'],r['cell_id']))
        if ordered:
            indices=list(dict.fromkeys(np.linspace(0,len(ordered)-1,min(count,len(ordered))).round().astype(int)))
            buckets[cat]=[ordered[i]['cell_id'] for i in indices]
    while len(selected)<count and any(buckets.values()):
        for bucket in buckets.values():
            if bucket and len(selected)<count:
                selected.append(bucket.pop(0))
    if len(selected)!=count: raise ValueError('insufficient eligible tiny TRAIN cells')
    return selected


def raw_b2c1(record, cell, root):
    path=Path(record['raw_source']); ref=record['reference_capacity_ah']
    with h5py.File(path,'r') as mat:
        group=mat['batch']; summary=mat[group['summary'][1,0]]; traces=mat[group['cycles'][1,0]]
        x=np.asarray(summary['cycle']).reshape(-1).astype(int)
        q=np.asarray(summary['QDischarge']).reshape(-1)
        rows,rejected=mit_prefix(mat,1,max_origin=100)
        extracted={cy:(mit_prefix_to_amperes(raw[None])[0] if cell.get('current_scale_to_ampere') else raw) for cy,raw in rows}
        deviations=[]
        for cy,raw in zip(cell['cycle'],cell['raw']):
            if cy>100: continue
            if int(cy) not in extracted: raise ValueError('b2c1 prepared prefix absent in raw')
            np.testing.assert_array_equal(np.isnan(raw),np.isnan(extracted[int(cy)]))
            deviations.append(float(np.nanmax(abs(raw-extracted[int(cy)]))))
        examples=[]; waves={}
        for cy in (2,20,50,100,101,170):
            indices=np.flatnonzero(x==cy)
            if not len(indices): continue
            j=int(indices[0]); t,v,i,temp,qd=[read_ref(mat,traces[name][j,0]) for name in ('t','V','I','T','Qd')]
            examples.append({'cycle':cy,'summary_discharge_ah':float(q[j]),
                'trace_qd_max_ah':float(np.max(qd)), 'summary_trace_difference_ah':float(abs(q[j]-np.max(qd))),
                'reference_soh':float(q[j]/ref), 'samples':len(t),
                'minimum_time_step_seconds':float(np.min(np.diff(t))*60),
                'units':{'time':'minutes','raw_stored_current':'author-normalized C-rate',
                    'prepared_current':'A' if cell.get('current_scale_to_ampere') else 'legacy numeric C-rate mislabeled A',
                    'voltage':'V','temperature':'C','capacity':'Ah'}})
            if cy<=100:
                waves.update({f'{cy}__{name}':val for name,val in [('time_minutes',t),('voltage_v',v),('current_a',i),('temperature_c',temp)]})
        np.savez_compressed(root/'b2c1_original_prefix_traces.npz',**waves)
    valid=(cell['label_cycle']>0)&np.isfinite(cell['soh'])
    q_by_cycle=dict(zip(x,q)); label_error=max(abs(cell['soh'][j]-q_by_cycle[int(cy)]/ref) for j,cy in enumerate(cell['label_cycle']) if valid[j])
    return {'cell_id':'b2c1','raw_file':str(path),'raw_sha256':fingerprint(path),
        'reference_capacity_ah':ref,'reference_rule':record['reference_rule'],
        'reference_established_source_cycle':cell['reference_established_source_cycle'],
        'not_a_documented_batch1_continuation':True,
        'max_prepared_feature_reextraction_error':max(deviations),
        'max_summary_reference_label_error_soh':float(label_error),
        'current_units_version':'physical_amperes_v2' if cell.get('current_scale_to_ampere') else 'legacy_stored_v1',
        'current_nominal_amperes_per_c':MIT_NOMINAL_AMPERES_PER_C,
        'extraction_rejections':rejected,'original_examples':examples,
        'conclusion':'Retain fast-degrading cell; no feature/cycle/reference mismatch found by these checks. Raw Q used only in audit, never inference.',
        'future_example_scope':'label provenance check only; excluded from TRAIN selection and fitting'}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/llm_curve_repair_v1.json'); args=ap.parse_args()
    settings=json.loads(Path(args.config).read_text(encoding='utf-8'))
    manifest,cells,plan,registry_path,prepared=load_frozen(settings)
    root=Path('experiments/llm_curve_repair')/('P0_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    root.mkdir(parents=True,exist_ok=False);print('RUN='+str(root.resolve()),flush=True)
    rows=training_inventory(cells,plan['train'],settings['origin']); selected=select_tiny(rows,settings['tiny_cells'])
    save_json(root/'config.json',{**settings,'plan':plan,'prepared_folder':str(prepared),'tiny_train_ids':selected})
    save_json(root/'split_manifest.json',{'plan':plan,'physical_family':cells[plan['train'][0]]['physical_family'],
        'tiny_train_ids':selected,'tiny_selection_rules':'TRAIN-only category round-robin over rate quantiles; no validation/heldout label selection',
        'historically_viewed':['MIT','XJTU'],'evidence_identity':'development-only','independent_confirmation_ids':[],
        'independent_confirmation_status':'unproven until data-use ledger for remaining datasets is audited',
        'author_documented_MIT_continuations_excluded':['b2c7','b2c8','b2c9','b2c15','b2c16']})
    save_json(root/'data_contract.json',{'origin_source_cycles':settings['origin'],'signals':'first 300s MIT main discharge V/I/T and derived time-prefix integrals',
        'raw_width':cells[plan['train'][0]]['raw'].shape[1], 'capacity_and_soh':'supervision/audit only; never model_input',
        'current_units_version':settings.get('mit_current_units','legacy_stored_v1'),
        'current_storage_convention':'Author MAT I is normalized by fixed nominal 1.1 A; legacy feature names incorrectly imply A',
        'current_conversion_source':'https://github.com/chueh-ermon/BMS-autoanalysis/blob/master/cell_analysis.m',
        'soh_reference':'initial accepted comparable discharge Ah; not nominal SOH',
        'preprocessing_fit_ids':plan['train'],'tiny_fit_ids':selected,'known_conditions_availability':'published fixed protocol before cycling; no realized future protocol',
        'prefix_source_cycle_limit':manifest['source_cycle_limit'],'longer_budget_requires_reextraction':manifest['source_cycle_limit']<200})
    save_json(root/'TRAIN_inventory.json',{'records':rows,'summary':{'cells':len(rows),
        'categories':dict(Counter(r.get('category','ineligible') for r in rows)),
        'origin_labels_available':sum(r.get('origin_label_available',False) for r in rows)}})
    record=next(r for r in manifest['cells'] if r['cell_id']=='b2c1')
    diagnosis=raw_b2c1(record,cells['b2c1'],root);save_json(root/'diagnosis.json',diagnosis)
    save_json(root/'failure_registry.json',{'excluded_by_prediction_error':[], 'current_goal_complete':False,
        'required_remaining':['P1 tiny end-to-end fit','P2 validation ablations','P3 all five datasets and five seeds','P4 strict all-cell acceptance and independent confirmation']})
    sources=[Path(__file__),Path(args.config),Path('docs/LLM_SMETimes_curve_repair_goal_20261011.md'),
        Path('battery_smetimes/research_llm.py'),Path('battery_smetimes/research_sequences.py'),Path('battery_smetimes/mit_units.py'),
        Path('battery_smetimes/llm_curve_objective.py'),Path('scripts/train_same_type_forecast.py'),
        Path('scripts/prepare_same_type_forecast.py'),Path('prepare_mit.py'),registry_path,
        Path(settings['objective_file'])]
    hashes={}
    for file in sources:
        target=root/'source_snapshot'/file.name
        if target.exists(): target=root/'source_snapshot'/('original_'+file.name)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,target)
        hashes[str(file.resolve())]=fingerprint(file)
    hashes[str((prepared/'manifest.json').resolve())]=fingerprint(prepared/'manifest.json')
    save_json(root/'source_manifest.json',{'source_sha256':hashes,
        'prepared_cells_sha256':{r['cell_id']:r['sha256'] for r in manifest['cells']},
        'checkpoint_sha256':{str(p):fingerprint(p) for p in Path(settings['checkpoint']).iterdir() if p.suffix in ('.json','.safetensors')}})
    save_json(root/'status.json',{'P0_source_checks_complete':True,'P0_complete':False,
        'independent_confirmation_pending':True,'training_started':False,'goal_complete':False})
    save_json(Path('experiments/llm_curve_repair/latest_P0.json'),{'folder':str(root.resolve())})
    print(json.dumps({'tiny_ids':selected,'raw_diagnosis':diagnosis,'output':str(root.resolve())},ensure_ascii=False),flush=True)


if __name__=='__main__': main()
