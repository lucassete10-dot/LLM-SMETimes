"""Independent NCA inner-CV audit, original outer holdout unchanged."""
import argparse,json
from pathlib import Path
import joblib
import numpy as np
from battery_smetimes import continuous_length as point
from scripts.train_continuous_length_pilot import load_nominal,target
from scripts.verify_condition_length import check_base
from scripts.prepare_same_type_forecast import sha,save_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);root=ap.parse_args().run
    cfg=json.loads((root/'config.json').read_text());status=json.loads((root/'progress.json').read_text())
    assert status['terminal'] and status['success']
    for p,d in cfg['source_sha256'].items():assert sha(Path(p))==d
    original=cfg['original_plan'];dev=cfg['development_cells'];outer=original['test']
    assert not set(dev)&set(outer)
    if cfg.get('full_early_labels_loaded'):
        from scripts.nca_full_label_loader import load
        assert sha(Path(cfg['label_index']))==cfg['label_index_sha256']
        cells,_=load(dict(original,train=dev,validation=[]),cfg['label_index'])
    else:cells,_=load_nominal(dict(original,train=dev,validation=[]))
    assert set(cells)==set(dev)
    assert sorted(k for plan in cfg['plans'] for k in plan['validation'])==sorted(dev)
    models=curves=metrics=0
    for plan in cfg['plans']:
        assert plan['test']==outer
        assert not set(plan['train'])&set(plan['validation']+outer)
        assert set(plan['train']+plan['validation'])==set(dev)
        for origin in cfg['origins']:
            targets={k:target(cells[k],origin) for k in dev}
            for seed in cfg['seeds']:
                folder=root/plan['name']/str(origin)/str(seed);rec=json.loads((folder/'config.json').read_text())
                path=folder/'model.joblib';assert sha(path)==rec['model_sha256']
                b=joblib.load(path);assert b['origin']==origin and b['seed']==seed
                assert rec['train']==plan['train'] and rec['validation']==plan['validation']
                check_base(b,cells,targets,plan['train'])
                rows=json.loads((folder/'metrics.json').read_text())
                with np.load(folder/'predictions.npz',allow_pickle=False) as data:
                    assert set(data.files)=={k+'__'+n for k in plan['validation'] for n in ('cycle','truth','prediction')}
                    for cid in plan['validation']:
                        c,y=targets[cid];p=point.predict(b,cells[cid],c)
                        if cfg.get('full_early_labels_loaded') and origin<100:
                            assert np.any((c>origin)&(c<=100)), 'Missing actual short-input evaluation interval'
                        for n,v in [('cycle',c),('truth',y),('prediction',p)]:np.testing.assert_allclose(data[cid+'__'+n],v,atol=1e-12,rtol=0)
                        r=[row for row in rows if row['cell_id']==cid]
                        assert {row['horizon'] for row in r}>={'common','operational'}
                        for row in r:
                            horizon=row['horizon']
                            mask=np.ones(len(c),dtype=bool) if horizon=='operational' else (c>100 if horizon=='common' else c<=origin+int(horizon.split('_')[1]))
                            e=p[mask]-y[mask];cy=c[mask];late=cy>=cy[0]+.8*(cy[-1]-cy[0])
                            vals=dict(mape_percent=100*np.mean(abs(e)/y[mask]),mae_soh=np.mean(abs(e)),rmse_soh=np.sqrt(np.mean(e**2)),late20_mae_pp=100*np.mean(abs(e[late])),endpoint_pp=100*abs(e[-1]),drop_error_pp=100*abs(e[0]-e[-1]))
                            yy,pp=y[mask],p[mask]
                            # Independent stage medians, using explicit stable distance ordering.
                            levels=[]
                            for values in (yy,pp):
                                levels.append([float(np.median(values[sorted(range(len(cy)),key=lambda j:(abs(cy[j]-boundary),j))[:min(5,len(cy))]])) for boundary in np.linspace(cy[0],cy[-1],4)])
                            yd=-100*np.diff(levels[0]);pd=-100*np.diff(levels[1])
                            vals['stage_error_pp']=float(max(abs(yd-pd)))
                            vals['fade_normalized_mae']=float(100*np.mean(abs(e))/max(abs(100*(yy[0]-yy[-1])),2))
                            assert row['acceleration_direction_reversed']==bool(yd[-1]-yd[0]>1 and pd[-1]-pd[0]<-1)
                            assert row['short_coverage']==bool(len(cy)<10 or cy[-1]-cy[0]<100)
                            total=np.sum((yy-yy.mean())**2)
                            if total<=1e-12:assert row['r2'] is None
                            else:vals['r2']=float(1-np.sum(e**2)/total)
                            for n,v in vals.items():np.testing.assert_allclose(row[n],v,atol=1e-10,rtol=0)
                            if horizon.startswith('fixed_'):assert row['full_horizon_covered']==bool(c[-1]>=origin+int(horizon.split('_')[1]))
                            metrics+=1
                        curves+=1
                models+=1
    assert models==status['completed_fits']==150
    assert curves==len(dev)*len(cfg['origins'])*len(cfg['seeds'])
    proof=dict(passed=True,verified_models=models,verified_curves=curves,verified_metric_rows=metrics,
        whole_cell_partitions=True,fixed_outer_excluded=True,all_development_cells_held_once_per_seed_length=True,
        train_preprocessing_rebuilt=True,saved_weight_replay=True,independent_basic_metrics=True,
        independent_stage_and_reversal_metrics=True,
        two_named_horizons_replayed=True,operational_coverage_verified=bool(cfg.get('full_early_labels_loaded')),
        cached_label_minimum=float(min(target(cells[k],0)[0][0] for k in dev)),
        coverage_correction='Recovered early labels included; observed gaps retained.' if cfg.get('full_early_labels_loaded') else 'Legacy NCA cache stores only cycle>100 labels. Named operational rows do not validate N..100 for N<100. Common-after100 metrics remain valid.',
        outer_loaded=False,goal_complete=False,
        limitation='Development selection only; stage/knee acceptance and genuinely untouched status of historical outer cells not proven.')
    save_json(root/'verification.json',proof);print(json.dumps(proof))


if __name__=='__main__':main()
