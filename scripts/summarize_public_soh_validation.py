"""Recompute all frozen held-out metrics and tail coverage without refitting."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import traceback
import csv
import joblib
import numpy as np
from battery_smetimes.external_soh import load_context
from battery_smetimes.research_baselines import regression_metrics
from scripts.prepare_same_type_forecast import save_json,sha

TREES=Path('experiments/stm32_compact_trees/runs/20261004T074429_087457Z')
NASA=Path('experiments/nasa_signal_trajectory/ridge/20261004T143636_110096Z')
OXFORD=Path('experiments/oxford_sparse_diagnostic/20261004T145450_603603Z')
RELEASE=Path('deployment/stm32_release_trees/20261004T083706_476897Z')
NCA=Path('experiments/zhu2022_nca_forecast/20261004T161453_449950Z')
NCA_VERIFIED=Path('experiments/zhu2022_nca_verification/20261004T163114_785206Z/verification.json')
METRICS=['mape_percent','mae_soh','rmse_soh','r2']

def recompute(dataset,model,run,plans,seeds,cells,nested):
    rows=[]
    for plan in plans:
        for seed in seeds:
            folder=run/plan['name']/str(seed)
            dest=folder/'test' if nested else folder
            report=json.loads((dest/'metrics.json').read_text())
            reported=report['by_cell'] if isinstance(report,dict) else report
            assert {r['cell_id'] for r in reported}==set(plan['test'])
            if isinstance(report,dict):assert not report['skipped']
            bundle=joblib.load(folder/'model.joblib')
            assert set(bundle['train_ids'])==set(plan['train']+plan['validation'])
            assert not set(bundle['train_ids'])&set(plan['test'])
            with np.load(dest/'predictions.npz',allow_pickle=False) as archive:
                for row in reported:
                    key=row['cell_id'];cell=cells[key];prefix=key+'__100__'
                    x,y,p=[archive[prefix+f] for f in ('cycle','truth','prediction')]
                    assert np.all(np.diff(x)>0) and np.all(x>100) and np.isfinite(p).all()
                    if dataset in ('Oxford','Zhu2022_NCA'):expected_x,expected_y=cell['cycle'],cell['truth']
                    else:
                        take=np.isfinite(cell['soh'])&(cell['soh']>0)&(cell['label_cycle']>100)
                        expected_x,expected_y=cell['label_cycle'][take],cell['soh'][take]
                    np.testing.assert_array_equal(x,expected_x);np.testing.assert_array_equal(y,expected_y)
                    full=regression_metrics(y,p)
                    for metric in METRICS:
                        if full[metric] is None:assert row['metrics'][metric] is None
                        else:np.testing.assert_allclose(full[metric],row['metrics'][metric],rtol=0,atol=1e-12)
                    late=x>=(x[0]+x[-1])/2;last=x>x[-1]-100
                    rows.append(dict(dataset=dataset,model=model,cell_id=key,fold=plan['name'],seed=seed,
                        metrics=full,late_half=regression_metrics(y[late],p[late]),
                        last100=regression_metrics(y[last],p[last]),
                        endpoint_abs_soh_pp=float(abs(y[-1]-p[-1])*100),future_targets=len(y),
                        future_source_cycle_first=float(x[0]),future_source_cycle_last=float(x[-1]),
                        last100_window_lower_exclusive=float(x[-1]-100),
                        last100_observed_targets=int(last.sum()),last100_observed_span=float(np.ptp(x[last])),
                        model_sha256=sha(folder/'model.joblib'),prediction_sha256=sha(dest/'predictions.npz'),
                        prediction_file=str((dest/'predictions.npz').resolve())))
    expected=sorted(k for k,c in cells.items() if dataset in ('Oxford','Zhu2022_NCA') or c['metadata']['dataset']==dataset)
    for seed in seeds:assert sorted(r['cell_id'] for r in rows if r['seed']==seed)==expected
    return rows

def aggregate(rows):
    seeds=sorted({r['seed'] for r in rows});summary={}
    def defined_mean(values):
        defined=[v for v in values if v is not None]
        return float(np.mean(defined)) if defined else None
    def reduce(values):
        defined=[v for v in values if v is not None]
        return dict(mean=defined_mean(values),seed_sd=float(np.std(defined,ddof=1)) if len(defined)>1 else None,
            seed_values=values,defined_seed_count=len(defined))
    for section in ('metrics','late_half','last100'):
        summary[section]={}
        for metric in METRICS:
            values=[defined_mean([r[section][metric] for r in rows if r['seed']==s]) for s in seeds]
            summary[section][metric]=reduce(values)
            summary[section][metric]['defined_cell_seed_count']=sum(r[section][metric] is not None for r in rows)
    summary['endpoint_abs_soh_pp']=reduce([np.mean([r['endpoint_abs_soh_pp'] for r in rows if r['seed']==s]) for s in seeds])
    ids=sorted({r['cell_id'] for r in rows});cell_rows=[]
    for key in ids:
        subset=[r for r in rows if r['cell_id']==key]
        means={metric:defined_mean([r['metrics'][metric] for r in subset]) for metric in METRICS}
        cell_rows.append(dict(cell_id=key,metrics_mean=means,
            mape_seed_sd=float(np.std([r['metrics']['mape_percent'] for r in subset],ddof=1)),
            max_endpoint_abs_soh_pp=float(max(r['endpoint_abs_soh_pp'] for r in subset))))
    failures=[r for r in cell_rows if r['metrics_mean']['mape_percent']>6]
    summary.update(cells=len(ids),cell_seed_evaluations=len(rows),seeds=seeds,
        fraction_cells_mean_mape_le6=(len(ids)-len(failures))/len(ids),
        failures_by_mean_mape=failures,by_cell_seed_aggregates=cell_rows,
        worst_cell=max(cell_rows,key=lambda r:r['metrics_mean']['mape_percent']),
        cohort_mean_gate_passed=summary['metrics']['mape_percent']['mean']<=6,
        independent_blind_test=False,role='public-development whole-cell validation',
        averaging='first equal-cell metric mean within each seed, then mean and sample SD across seeds; no row pooling')
    return summary

def main():
    out=Path('results/public_soh_validation')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True)
    shutil.copy2(__file__,out/Path(__file__).name)
    try:
        context=load_context(RELEASE);cells=context['cells']
        plans=json.loads((RELEASE/'frozen_audit/splits.json').read_text())['plans']
        seeds=json.loads((TREES/'config.json').read_text())['seeds'];all_rows=[];summaries={}
        for domain,model,run in [('MIT','ExtraTrees',TREES),('XJTU','ExtraTrees',TREES),('NASA','Ridge',NASA)]:
            rows=recompute(domain,model,run,[p for p in plans if p['dataset']==domain],seeds,cells,run==TREES)
            all_rows.extend(rows);summaries[domain]=aggregate(rows)
            summaries[domain]['deterministic_repetitions']=domain=='NASA'
        cfg=json.loads((OXFORD/'config.json').read_text());oxcells=joblib.load(OXFORD/'prepared_cells.joblib')
        rows=recompute('Oxford','Ridge_sparse_diagnostic',OXFORD,cfg['plans'],cfg['seeds'],oxcells,False)
        all_rows.extend(rows);summaries['Oxford']=aggregate(rows)
        summaries['Oxford'].update(deterministic_repetitions=True,
            task='sparse diagnostics at source cycles0/100; known protocol I, not dense measured I')
        verified=json.loads(NCA_VERIFIED.read_text());assert verified['passed']
        assert Path(verified['run']).resolve()==NCA.resolve()
        cfg=json.loads((NCA/'config.json').read_text())
        prepared=Path(cfg['prepared_directory'])
        assert sha(prepared/'prepared_cells.joblib')==cfg['prepared_sha256']
        ncacells=joblib.load(prepared/'prepared_cells.joblib')
        for name,label in [('ridge','Ridge'),('extra_trees','ExtraTrees')]:
            rows=recompute('Zhu2022_NCA',label,NCA/name,cfg['plans'],cfg['seeds'],ncacells,False)
            all_rows.extend(rows);summaries['Zhu2022_NCA_'+label]=aggregate(rows)
            summaries['Zhu2022_NCA_'+label].update(deterministic_repetitions=name=='ridge',
                raw_cells=66,eligible_cells=55,task_ineligible_short=10,structural_counter_reset=1,
                task='measured V/I from first100 source cycles; independent physical cohort')
        save_json(out/'by_cell.json',all_rows);save_json(out/'summary.json',summaries)
        unique_cells={(r['dataset'],r['cell_id']) for r in all_rows}
        flat=[]
        for row in all_rows:
            item={k:row[k] for k in ('dataset','model','cell_id','fold','seed','future_targets','endpoint_abs_soh_pp')}
            for section in ('metrics','late_half','last100'):
                item.update({section+'_'+key:value for key,value in row[section].items()})
            flat.append(item)
        with (out/'summary.csv').open('w',encoding='utf-8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
        save_json(out/'manifest.json',dict(complete=True,cells=len(unique_cells),cell_seed_evaluations=len(all_rows),
            recomputed_metrics=True,full_valid_future_label_coverage=True,development_train_test_disjoint=True,
            source_sha256=sha(__file__),source_runs=[str(p.resolve()) for p in (TREES,NASA,OXFORD,NCA)],
            nca_verification_sha256=sha(NCA_VERIFIED),alternative_model_arms_not_extra_physical_cells=True,
            deterministic_zero_sd_is_not_independent_replication=True,all_cell_failures_retained=True))
        lines=['# Public SOH held-out development summary','','All metrics: equal-cell macro means. SOH scale0–1; MAPE%. SD is across5 seeds, not confidence in unseen populations. NASA/Oxford/NCA Ridge are deterministic repetitions. Oxford is a different sparse diagnostic task. NCA model arms share55 physical cells and are not110 additional cells. Low MAPE does not establish accurate curve shape; negative mean R² and endpoint failures remain reported.','',
            '| Dataset | Cells | MAPE% ± seedSD | MAE | RMSE | mean R² | Cells mean MAPE≤6% | Late-half MAPE% | Last100 MAPE% | Endpoint SOHpp |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for d,s in summaries.items():
            m=s['metrics'];a=m['mape_percent'];lines.append(f'| {d} | {s["cells"]} | {a["mean"]:.3f} ± {a["seed_sd"]:.3f} | {m["mae_soh"]["mean"]:.5f} | {m["rmse_soh"]["mean"]:.5f} | {m["r2"]["mean"]:.3f} | {s["fraction_cells_mean_mape_le6"]:.1%} | {s["late_half"]["mape_percent"]["mean"]:.3f} | {s["last100"]["mape_percent"]["mean"]:.3f} | {s["endpoint_abs_soh_pp"]["mean"]:.3f} |')
        # Generated experiment report, not a manual source edit.
        (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print('OUTPUT='+str(out.resolve()),'evaluations='+str(len(all_rows)),flush=True)
    except BaseException:
        save_json(out/'failure.json',dict(traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
