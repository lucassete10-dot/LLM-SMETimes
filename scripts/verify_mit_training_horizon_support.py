"""Independent replay of all train-side horizon counts and observed labels."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from scripts.train_curve_fidelity import load_subset
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.verify_nasa_robust_inner import close


def actual(cell):
    take=(cell['label_cycle']>100)&np.isfinite(cell['soh'])&(cell['soh']>0)
    return cell['label_cycle'][take],cell['soh'][take]


def recalc(cells,ids,h):
    ending=[];values=[]
    for k in ids:
        c,y=actual(cells[k]);ending.append(float(c[-1]))
        if c[-1]<h:continue
        j=min(range(len(c)),key=lambda j:abs(float(c[j])-h))
        if abs(float(c[j])-h)<=25:values.append(float(y[j]))
    covered=sum(e>=h for e in ending)
    return dict(training_cells=len(ids),cells_observed_to_horizon=covered,
        observed_to_horizon_fraction=covered/len(ids),actual_label_count=len(values),
        actual_soh_median=float(np.median(values)) if values else None,
        actual_soh_mean=sum(values)/len(values) if values else None,training_max_source_cycle=max(ending))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);root=ap.parse_args().run
    cfg=json.loads((root/'config.json').read_text());parent=Path(cfg['parent'])
    assert not cfg['outer_read'] and cfg['new_fits']==0
    for f,h in {**cfg['source_sha256'],**cfg['input_sha256']}.items():assert sha(f)==h
    assert sha(parent/'config.json')==cfg['parent_config_sha256'] and sha(parent/'independent_verification.json')==cfg['parent_proof_sha256']
    pc=json.loads((parent/'config.json').read_text());rows=json.loads((root/'validation_support.json').read_text())
    grid=json.loads((root/'training_horizon_grid.json').read_text());assert len(rows)==124 and len(grid)==40
    assert len({(r['fold'],r['cell_id']) for r in rows})==124
    for plan in pc['plans']:
        cells=load_subset(plan['train']+plan['validation']);assert not set(cells)&set(plan['test'])
        for k,v in cells.items():assert v['prepared_sha256']==cfg['prepared_sha256'][k]
        child=parent/plan['name']/str(cfg['seed']);bundle=joblib.load(child/'feature.joblib')
        selection=json.loads((child/'selection.json').read_text())
        groups={}
        for key in plan['train']:
            m=cells[key]['metadata'];group=((m.get('charge_protocol') or '').replace('-newstructure',''),m.get('diagnostic_discharge_rate_c'),m.get('discharge_cutoff_v'))
            groups.setdefault(group,[]).append(key)
        assert set(bundle['local'])=={k for k,v in groups.items() if len(v)>=6}
        for r in [r for r in rows if r['fold']==plan['name']]:
            assert r['cell_id'] in plan['validation'];m=cells[r['cell_id']]['metadata']
            group=((m.get('charge_protocol') or '').replace('-newstructure',''),m.get('diagnostic_discharge_rate_c'),m.get('discharge_cutoff_v'))
            ids=groups[group] if group in bundle['local'] else plan['train']
            c,y=actual(cells[r['cell_id']]);assert r['query_end_source_cycle']==c[-1] and r['validation_endpoint_soh']==y[-1]
            assert r['route']==('local' if group in bundle['local'] else 'family')
            assert r['selected_new_weight']==selection['selected']['new_weight']
            for k,v in recalc(cells,ids,float(c[-1])).items():close({k:v},{k:r[k]})
        for r in [r for r in grid if r['fold']==plan['name']]:
            assert r['source_cycle'] in range(200,1601,200)
            for k,v in recalc(cells,plan['train'],r['source_cycle']).items():close({k:v},{k:r[k]})
    summary=json.loads((root/'summary.json').read_text())
    for label,subset in [('all_validation_records',rows),('selected_new_component_records',[r for r in rows if r['selected_new_weight']>0])]:
        close(dict(count=len(subset),mean_observed_to_horizon_fraction=float(np.mean([r['observed_to_horizon_fraction'] for r in subset])),
            zero_support_count=sum(r['cells_observed_to_horizon']==0 for r in subset),at_most_two_support_count=sum(r['cells_observed_to_horizon']<=2 for r in subset)),summary[label])
    save_json(root/'independent_verification.json',dict(passed=True,validation_records=124,training_grid_rows=40,
        outer_read=False,new_fits=0,verifier_sha256=sha(__file__),scope='counts/real observations/route/metadata/SHA; no causal claim',goal_complete=False))
    print('PASSED 124 validation support and 40 training grid records',flush=True)


if __name__=='__main__':main()
