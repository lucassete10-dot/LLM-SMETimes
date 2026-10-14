"""Independent stored-array audit of the inventory and train-only support counts."""
import argparse,json
from pathlib import Path
from collections import Counter
import joblib,numpy as np
from scripts.prepare_same_type_forecast import save_json,sha
from scripts.train_curve_fidelity import AUDIT

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);root=ap.parse_args().run
    cfg=json.loads((root/'config.json').read_text());report=json.loads((root/'coverage.json').read_text())
    for f,d in cfg['source_sha256'].items():assert sha(f)==d==sha(root/'source_snapshot'/Path(f).name)
    records=json.loads((AUDIT/'cell_registry.json').read_text())['cells'];assert sha(AUDIT/'cell_registry.json')==cfg['cell_registry_sha256']
    assert sha(AUDIT/'source_snapshot/battery_physical_registry.json')==cfg['physical_registry_sha256']
    acfg=json.loads((AUDIT/'config.json').read_text());issues=json.loads(Path(acfg['config']['integrity_audit']).read_text())['issues']
    assert sha(acfg['config']['integrity_audit'])==acfg['integrity_sha256']
    old=Path('experiments/zhu2022_nca_forecast/20261004T161453_449950Z/config.json');assert sha(old)==cfg['nca_config_sha256'];ncfg=json.loads(old.read_text());file=Path(ncfg['prepared_directory'])/'prepared_cells.joblib';assert sha(file)==ncfg['prepared_sha256'];nca=joblib.load(file)
    assert len(report['cells'])==248 and len({r['cell_id'] for r in report['cells']})==248
    lookup={r['cell_id']:r for r in report['cells']}
    assert set(lookup)==set(records)|set(nca)
    widths={'MIT':51,'XJTU':46,'NASA':44,'Zhu2022NCA':32}
    for r in report['cells']:
        key=r['cell_id'];d=r['dataset']
        if key in records:
            rec=records[key];p=Path(rec['prepared_file']);assert sha(p)==rec['prepared_sha256']==cfg['prepared_sha256'][str(p)]
            # Only signal keys are inspected; SOH/capacity arrays are not read.
            with np.load(p) as a:c,x=a['cycle'],a['raw_features']
            bad=[i['cycle'] for i in issues if i['cell_id']==key];take=(c<=100)&~np.isin(c,bad);c,x=c[take],x[take]
            assert r['physical_family']==rec['physical_family']
            if d=='MIT':assert 'batch' not in r['condition_stratum'] and 'charge_protocol' in r['condition_stratum']
        else:
            cell=nca[key];c,x=cell['early_cycle'],cell['early_raw'];take=c<=100;c,x=c[take],x[take]
        assert x.shape==(r['early_records'],widths[d]) and len(c)==r['early_records']
        assert r['first_source_cycle']==float(c[0]) and r['last_source_cycle']==float(c[-1])
        assert r['strictly_increasing']==bool(np.isfinite(c).all() and np.all(np.diff(c)>0))
        assert r['inf_values']==int(np.isinf(x).sum()) and r['nan_values']==int(np.isnan(x).sum())
        finite=np.isfinite(x);assert r['all_missing_columns']==[i for i in range(x.shape[1]) if not finite[:,i].any()]
        for channel,group in r['groups'].items():
            indices=[i for i,name in enumerate(report['schemas'][d]) if channel in name.lower()];assert group['columns']==indices
            z=finite[:,indices];actual=float(z.mean()) if z.size else None
            assert group['finite_fraction']==actual
            assert group['completely_missing_columns']==sum(not finite[:,i].any() for i in indices)
        assert r['capacity_labels_used'] is False and r['future_signals_used'] is False
    plans=json.loads((AUDIT/'splits.json').read_text())['plans']+[dict(p,dataset='Zhu2022NCA') for p in ncfg['plans']]
    assert len(plans)==len(report['train_only_local_support'])
    for p,s in zip(plans,report['train_only_local_support']):
        assert s['fold']==p['name'];counts=Counter(lookup[k]['condition_stratum'] for k in p['train']);assert dict(counts)==s['train_counts']
        assert s['eligible_local_strata']==[k for k,v in counts.items() if v>=6]
        assert s['fallback_strata']==[k for k,v in counts.items() if v<6]
        assert not set(p['train'])&set(p['validation']) and not set(p['train']+p['validation'])&set(p['test'])
    for d,s in report['summary'].items():
        rr=[r for r in report['cells'] if r['dataset']==d];assert len(rr)==s['cells']
        assert s['early_records_range']==[min(r['early_records'] for r in rr),max(r['early_records'] for r in rr)]
        assert s['conditions']==dict(Counter(r['condition_stratum'] for r in rr))
        assert s['physical_families']==dict(Counter(r['physical_family'] for r in rr))
        assert s['cells_with_missing_temperature']==sum(r['groups']['temperature']['finite_fraction'] in (None,0.) for r in rr)
    assert report['fitted_statistics'] is False and report['model_or_feature_selection'] is False and not report['new_exclusions']
    save_json(root/'independent_verification.json',dict(passed=True,cells=248,folds=len(plans),signal_array_recomputed=True,source_sha256=sha(__file__),no_new_exclusions=True,goal_complete=False))
    print('VERIFIED 248 cells, '+str(len(plans))+' frozen fold support counts',flush=True)

if __name__=='__main__':main()
