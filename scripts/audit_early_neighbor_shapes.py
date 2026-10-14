"""Training-only early-feature/late-shape association, not model evaluation."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from battery_smetimes.conditional_drift import prefix
from battery_smetimes.tail_boost import labels,conditions
from scripts.train_tail_boost import load_cells
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import sha,save_json

def shape_distance(ca,ya,cb,yb):
    lo=max(ca[0],cb[0]);hi=min(ca[-1],cb[-1])
    if hi<=lo:return None
    coords=np.linspace(lo,hi,33);a=np.interp(coords,ca,ya);b=np.interp(coords,cb,yb)
    drop_a=100*(a[0]-a[-1]);drop_b=100*(b[0]-b[-1])
    accel_a=100*(2*a[16]-a[0]-a[-1]);accel_b=100*(2*b[16]-b[0]-b[-1])
    return dict(common_start=float(lo),common_end=float(hi),common_span=float(hi-lo),mean_abs_soh_difference_pp=float(100*np.mean(np.abs(a-b))),drop_difference_pp=float(abs(drop_a-drop_b)),acceleration_difference_pp=float(abs(accel_a-accel_b)))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);args=parser.parse_args();root=args.source;cfg=json.loads((root/'config.json').read_text());rows=[];summaries={};evidence=[]
    assert json.loads((root/'independent_verification.json').read_text())['passed']
    for plan in cfg['plans']:
        # Source has one frozen inner fold per cohort. Never load validation/test.
        cells=load_cells(dict(plan,validation=[]));ids=plan['train'];assert set(cells)==set(ids)
        assert len({cells[k]['physical_family'] for k in ids})==1
        raw=np.stack([prefix(clean(cells[k])) for k in ids]);imputer=SimpleImputer(strategy='median',keep_empty_features=True);x=imputer.fit_transform(raw);scaler=StandardScaler();z=scaler.fit_transform(x)
        curves={k:labels(cells[k]) for k in ids};pairs=[]
        for i,a in enumerate(ids):
            for j in range(i+1,len(ids)):
                b=ids[j];metrics=shape_distance(*curves[a],*curves[b])
                if metrics is None:continue
                pairs.append(dict(a=a,b=b,distance=float(np.linalg.norm(z[i]-z[j])),same_condition=conditions(cells[a])==conditions(cells[b]),**metrics))
        local=[]
        for key in ids:
            family=[p for p in pairs if key in (p['a'],p['b'])]
            for group,pool in [('family',family),('same_condition',[p for p in family if p['same_condition']])]:
                if not pool:continue
                near=min(pool,key=lambda p:(p['distance'],p['a'],p['b']))
                # Uniform peer mean is deterministic reference; no random lucky neighbor.
                fields=('mean_abs_soh_difference_pp','drop_difference_pp','acceleration_difference_pp')
                local.append(dict(cell_id=key,group=group,neighbor=near['b'] if near['a']==key else near['a'],peer_count=len(pool),feature_distance=near['distance'],common_span=near['common_span'],nearest={f:near[f] for f in fields},uniform_peer_mean={f:float(np.mean([p[f] for p in pool])) for f in fields}))
        summary={}
        for group in ('family','same_condition'):
            r=[p for p in local if p['group']==group];metrics=('mean_abs_soh_difference_pp','drop_difference_pp','acceleration_difference_pp')
            summary[group]=dict(covered_cells=len(r),cells=len(ids),informative_cells=sum(p['peer_count']>1 for p in r),nearest={f:float(np.mean([p['nearest'][f] for p in r])) if r else None for f in metrics},uniform_peer={f:float(np.mean([p['uniform_peer_mean'][f] for p in r])) if r else None for f in metrics})
        distances=[p['distance'] for p in pairs];targets=[p['mean_abs_soh_difference_pp'] for p in pairs]
        rho=float(spearmanr(distances,targets).statistic) if len(pairs)>2 and np.std(distances)>0 and np.std(targets)>0 else None
        summary['pair_distance_shape_spearman']=rho;summary['pair_count']=len(pairs);summaries[plan['dataset']]=summary
        rows.append(dict(dataset=plan['dataset'],plan=plan['name'],training_ids=ids,neighbors=local,pairs=pairs))
        evidence.append(dict(dataset=plan['dataset'],imputer_statistics=imputer.statistics_.tolist(),scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist()))
    data_hashes={k:cfg['prepared_sha256'][k] for p in cfg['plans'] for k in p['train']}
    save_json(root/'training_early_neighbor_shapes.json',dict(summary=summaries,rows=rows,preprocessing=evidence,prepared_sha256=data_hashes,source_config_sha256=sha(root/'config.json'),source_sha256=sha(__file__),loaded_train_only=True,validation_loaded=False,outer_loaded=False,model_selection=False,goal_complete=False,interpretation='Training-only associations using standardized existing prefix features. Pair curves compared only over shared observed source-cycle range; shorter common spans may mask later knees. Correlations are descriptive dependent pairs, not significance tests. Nearest peer uses early features only; no forecast accuracy or information-theoretic limit claimed.'))
    print(json.dumps(summaries),flush=True)

if __name__=='__main__':main()
