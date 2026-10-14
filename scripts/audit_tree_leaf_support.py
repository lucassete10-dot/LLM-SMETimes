"""Exact weighted forest donor reconstruction; validation-only diagnosis."""
import json, logging, shutil, time
from pathlib import Path
from datetime import datetime, timezone
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.shape_forecast import training_arrays,predict
from battery_smetimes.tail_boost import train_arrays,inputs,labels,conditions
from battery_smetimes.waveform_shape import matrix
from scripts.train_crossfit_drift_smoke import clean
from scripts.train_tail_boost import load_cells
from scripts.prepare_same_type_forecast import save_json,sha


def donors(train_leaves,query_leaf,sample_weight):
    out=np.zeros(len(sample_weight))
    for tree,leaf in enumerate(query_leaf):
        rows=train_leaves[:,tree]==leaf
        if not rows.any():raise AssertionError('query reached leaf with no training rows')
        out[rows]+=sample_weight[rows]/sample_weight[rows].sum()/len(query_leaf)
    np.testing.assert_allclose(out.sum(),1,atol=1e-12,rtol=0)
    return out


def main():
    source=Path('experiments/shape_xjtu_fivefold/20261005T082211_584949Z')
    cfg=json.loads((source/'config.json').read_text());assert cfg['fivefold_shape']
    assert json.loads((source/'independent_verification.json').read_text())['passed']
    root=Path('experiments/tree_leaf_support')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True);snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),*[Path('battery_smetimes')/(n+'.py') for n in ('shape_forecast','waveform_shape','tail_boost','forecast','research_baselines')]]
    for f in files:shutil.copy2(f,snap/f.name)
    prereg='docs/soh_leaf_support_audit_20261005.md'
    save_json(root/'config.json',dict(source_run=str(source.resolve()),source_config_sha256=sha(source/'config.json'),source_verification_sha256=sha(source/'independent_verification.json'),source_sha256={f.name:sha(f) for f in files},preregistration_sha256=sha(prereg),plans=cfg['plans'],no_model_training=True,outer_loaded=False))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(root/'audit.log',encoding='utf-8'),logging.StreamHandler()])
    reports=[];start=time.perf_counter();checked=0
    for plan in cfg['plans']:
        cells=load_cells(plan);ids=plan['train']
        assert set(cells)==set(ids+plan['validation']) and not set(cells)&set(plan['test'])
        source_cycles=[];cell_indices=[];horizons=[]
        for i,key in enumerate(ids):
            x,y=labels(cells[key]);idx=np.unique(np.linspace(0,len(y)-1,min(96,len(y))).round().astype(int))
            source_cycles.extend(x[idx]);cell_indices.extend([i]*len(idx));horizons.append(float(x[-1]))
        source_cycles=np.asarray(source_cycles);cell_indices=np.asarray(cell_indices);horizons=np.asarray(horizons)
        for candidate in (0,1):
            folder=source/plan['name']/('candidate_'+str(candidate));bundle=joblib.load(folder/'model.joblib')
            base=bundle['base'] if candidate==0 else bundle
            assert base['train_ids']==ids and base['seed']==cfg['seed'] and not base.get('local')
            x,y,w=train_arrays(cells,ids,3) if candidate==0 else training_arrays(cells,ids)
            assert len(y)==len(source_cycles)
            model=base['model'];imputer=base['imputer'];z=imputer.transform(x)
            leaves=model.apply(z);assert leaves.shape==(len(y),128)
            with np.load(folder/'validation_predictions.npz') as saved:
                for key in plan['validation']:
                    cc=clean(cells[key]);cycle,truth=labels(cells[key]);idx=np.unique(np.linspace(0,len(cycle)-1,min(16,len(cycle))).round().astype(int));q=cycle[idx]
                    qx=inputs(cc,q) if candidate==0 else matrix(cc,q,True)
                    qleaves=model.apply(imputer.transform(qx));p=predict(bundle,cc,q)
                    np.testing.assert_allclose(p,saved[key+'__100__prediction'][idx],atol=1e-12,rtol=0)
                    rows=[];weights=[]
                    compatible=np.array([conditions(cells[k])==conditions(cc) for k in ids])
                    for j,t in enumerate(q):
                        donor=donors(leaves,qleaves[j],w)
                        np.testing.assert_allclose(donor@y,p[j],atol=1e-12,rtol=0)
                        cw=np.bincount(cell_indices,weights=donor,minlength=len(ids));weights.append(cw)
                        active=cw>0
                        rows.append(dict(query_cycle=float(t),prediction=float(p[j]),truth=float(truth[idx[j]]),global_horizon=float(horizons.max()),known_protocol_horizon=float(horizons[compatible].max()) if compatible.any() else None,active_donor_horizon=float(horizons[active].max()),weight_on_shorter_cells=float(cw[horizons<t].sum()),weighted_donor_row_cycle=float(donor@source_cycles),donor_cells=int(active.sum()),effective_donor_cells=float(1/(cw@cw)),unchanged_leaves_from_previous=bool(j>0 and np.array_equal(qleaves[j],qleaves[j-1]))))
                    report=dict(fold=plan['name'],candidate=candidate,cell_id=key,model_sha256=sha(folder/'model.joblib'),source_predictions_sha256=sha(folder/'validation_predictions.npz'),points=rows,donor_ids=ids)
                    reports.append(report);out=root/plan['name']/('candidate_'+str(candidate));out.mkdir(parents=True,exist_ok=True)
                    save_json(out/(key+'_support.json'),report);np.savez_compressed(out/(key+'_donor_weights.npz'),cycle=q,cell_weight=np.stack(weights),training_horizon=horizons,training_cell_indices=cell_indices,training_row_cycle=source_cycles)
                    checked+=len(q)
            logging.info('PASS %s candidate=%s',plan['name'],candidate)
    save_json(root/'summary.json',reports)
    aggregate={}
    for candidate in (0,1):
        selected=[r for r in reports if r['candidate']==candidate];ends=[r['points'][-1] for r in selected]
        aggregate[str(candidate)]=dict(validation_cell_fold_rows=len(ends),endpoints_beyond_global=sum(r['query_cycle']>r['global_horizon'] for r in ends),endpoints_beyond_active_donor=sum(r['query_cycle']>r['active_donor_horizon'] for r in ends),mean_endpoint_weight_on_shorter_cells=float(np.mean([r['weight_on_shorter_cells'] for r in ends])),mean_endpoint_effective_donors=float(np.mean([r['effective_donor_cells'] for r in ends])),identical_leaf_adjacent_pairs=sum(p['unchanged_leaves_from_previous'] for r in selected for p in r['points'][1:]),adjacent_pairs=sum(len(r['points'])-1 for r in selected))
    save_json(root/'verification.json',dict(passed=True,exact_prediction_reconstructions=checked,aggregate=aggregate,seconds=time.perf_counter()-start,outer_loaded=False,goal_complete=False))
    print('OUTPUT='+str(root.resolve()),flush=True);print(json.dumps(aggregate),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
