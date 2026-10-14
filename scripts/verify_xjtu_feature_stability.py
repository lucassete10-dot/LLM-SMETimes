"""Independent polyfit/scipy Theil-Sen calculations on TRAIN-only early arrays."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.stats import theilslopes
from prepare_xjtu import FEATURE_NAMES
from scripts.train_curve_fidelity import load_subset
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.verify_nasa_robust_inner import close


def main(root):
    cfg=json.loads((root/'config.json').read_text());plan=cfg['plan'];cells=load_subset(plan['train']);report=json.loads((root/'audit.json').read_text())
    assert len(cells)==31 and not set(cells)&set(plan['test']+plan['validation'])
    for f,d in cfg['input_sha256'].items():assert sha(Path(f))==d
    for f,d in cfg['source_sha256'].items():assert sha(root/'source_snapshot'/Path(f).name)==d
    for k,d in cfg['prepared_sha256'].items():assert cells[k]['prepared_sha256']==d
    cols=[i for i,n in enumerate(FEATURE_NAMES) if any(s in n for s in ('voltage','current','temperature'))]
    assert cols==cfg['feature_columns'] and cfg['feature_names']==[FEATURE_NAMES[i] for i in cols]
    computed=[];groups={};empty=missing=constant=0;counts=[]
    with np.load(root/'early_signals.npz',allow_pickle=False) as a:
        assert set(a.files)=={k+'__'+f for k in plan['train'] for f in ('cycle','signals')}
        assert len(report['cells'])==31
        for pos,k in enumerate(plan['train']):
            cell=cells[k];mask=cell['cycle']<=100;c=cell['cycle'][mask].astype(float);x=cell['raw'][mask][:,cols];r=report['cells'][pos]
            np.testing.assert_array_equal(a[k+'__cycle'],c);np.testing.assert_array_equal(a[k+'__signals'],x)
            m=cell['metadata'];condition=[m.get('charge_protocol'),m.get('discharge_rate_c'),m.get('discharge_cutoff_v')];groups.setdefault(tuple(condition),[]).append(k)
            assert r['cell_id']==k and r['condition']==condition and r['early_records']==len(c);counts.append(len(c))
            assert r['first']==c[0] and r['last']==c[-1]
            quarters=[sum((c>lo)&(c<=hi)) for lo,hi in [(0,25),(25,50),(50,75),(75,100)]]
            assert r['quarter_counts']==quarters;empty+=sum(v==0 for v in quarters);assert len(r['fields'])==len(cols)
            for j,col in enumerate(cols):
                f=r['fields'][j];v=x[:,j];vv=v[np.isfinite(v)];is_constant=bool(len(vv)>0 and np.ptp(vv)<1e-12)
                valid=np.isfinite(v[-10:]);t=c[-10:][valid].astype(float);z=v[-10:][valid].astype(float);missing+=not len(vv);constant+=is_constant
                expected=dict(feature=FEATURE_NAMES[col],finite_fraction=float(np.isfinite(v).mean()),constant=is_constant,finite_points=len(t),span=None,ols=None,theil_sen=None,sensitivity=None,flag=False)
                if len(t)>=2:
                    ols=float(np.polyfit(t-t.mean(),z-z.mean(),1)[0]*100);ts=float(theilslopes(z,t)[0]*100);span=float(t[-1]-t[0]);den=max(abs(ols),abs(ts),float(np.ptp(z)*100/span),1e-12);sensitivity=abs(ols-ts)/den
                    expected.update(span=span,ols=ols,theil_sen=ts,sensitivity=sensitivity,flag=sensitivity>.5+1e-12);computed.append(expected)
                close(f,expected)
    summary=dict(cells=31,columns=len(cols),record_range=[min(counts),max(counts)],empty_quarters=int(empty),all_missing_cell_columns=int(missing),constant_cell_columns=int(constant),calculable_trends=len(computed),flagged_trends=sum(f['flag'] for f in computed),median_sensitivity=float(np.median([f['sensitivity'] for f in computed])),p90_sensitivity=float(np.percentile([f['sensitivity'] for f in computed],90)))
    close(report['summary'],summary);assert len(report['conditions'])==len(groups)
    for r in report['conditions']:assert r['members']==groups[tuple(r['condition'])] and r['count']==len(r['members']) and r['local_supported']==(r['count']>=6)
    for key in ('labels_used','validation_loaded','outer_loaded'):assert cfg[key]==report[key]==False
    assert cfg['new_exclusions']==report['new_exclusions']==[]
    save_json(root/'independent_verification.json',dict(passed=True,cells=31,column_records=31*len(cols),independent_ols='numpy.polyfit',independent_robust='scipy.stats.theilslopes',labels_used=False,outer_loaded=False,verifier_sha256=sha(Path(__file__))))
    print('PASS:31TRAIN earlyarrays/columns/missingness/independentOLS-TheilSen/groups/SHA;no labels or exclusions')


if __name__=='__main__':main(Path(sys.argv[1]))
