"""Cell-bootstrap sensitivity of existing inner validation selection only."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.prepare_same_type_forecast import sha,save_json

def scores(values,indices):
    sampled=values[:,indices]
    return sampled.mean(axis=1)+.25*np.quantile(sampled,.9,axis=1)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text());assert json.loads((root/'independent_verification.json').read_text())['passed'];rng=np.random.default_rng(20261005);results=[];evidence=[]
    for plan in cfg['plans']:
        folder=root/plan['name'];values=[];n=len(plan['validation']);assert n>1
        for index in range(len(cfg['candidates'])):
            file=folder/f'candidate_{index}'/'validation_metrics.json';rows=json.loads(file.read_text());assert [r['cell_id'] for r in rows]==plan['validation']
            values.append([(r['mape_percent']+r['late_mape']+r['endpoint_pp']+r['drop_error_pp'])/4 for r in rows]);evidence.append(dict(file=str(file.resolve()),sha256=sha(file)))
        values=np.asarray(values);original=scores(values,np.arange(n));selected=int(np.argmin(original));assert selected==json.loads((folder/'selection.json').read_text())['selected']['candidate']
        winners=[];gains=[]
        for _ in range(1000):
            indices=rng.integers(0,n,size=n);sample=scores(values,indices);winners.append(int(np.argmin(sample)));gains.append(float(sample[0]-sample[selected]))
        results.append(dict(fold=plan['name'],validation_cells=n,selected_candidate=selected,original_score_gain=float(original[0]-original[selected]),selected_retained_fraction=float(np.mean(np.asarray(winners)==selected)),candidate_selection_counts={str(i):winners.count(i) for i in range(len(values))},fixed_selected_gain_quantiles=[float(x) for x in np.quantile(gains,[.025,.5,.975])],fixed_selected_gain_positive_fraction=float(np.mean(np.asarray(gains)>0))))
    save_json(root/'validation_selection_stability.json',dict(results=results,seed=20261005,repeats=1000,source_sha256=sha(__file__),evidence=evidence,outer_loaded=False,models_refit=False,used_for_selection=False,goal_complete=False,interpretation='Descriptive cell-bootstrap sensitivity on the same validation cells, not new validation, unbiased generalization estimate or guaranteed confidence interval. Score includes P90, unstable at small n. Fixed selected candidates remain unchanged.'))
    print(json.dumps(results),flush=True)

if __name__=='__main__':main()
