"""Prove whether anchor translation changes degradation-rate errors."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);root=parser.parse_args().run;proof=root/'independent_verification.json';assert json.loads(proof.read_text())['passed']
    cfg=json.loads((root/'config.json').read_text());summary=json.loads((root/'summary.json').read_text());pairs=0;maxdrop=0.;maxvariation=0.;rows=[];fallback=[];local=[]
    for lock in cfg['children']:
        cc=json.loads((Path(lock['inner_run'])/'config.json').read_text());folder=root/cc['parent_plan']['name'];record=next(r for r in summary['records'] if r['fold']==folder.name);winner=record['selected']['candidate'];rr=json.loads((folder/f'candidate_{winner}_metrics.json').read_text());support={r['cell_id']:r for r in json.loads((folder/'support.json').read_text())}
        for r in rr:
            rows.append(r)
            (fallback if support[r['cell_id']]['family_fallback'] else local).append(r)
        for i in range(1,7):
            raw=json.loads((folder/f'candidate_{i}_metrics.json').read_text());shape=json.loads((folder/f'candidate_{i+6}_metrics.json').read_text());assert [r['cell_id'] for r in raw]==[r['cell_id'] for r in shape]
            for a,b in zip(raw,shape):
                difference=abs(a['drop_error_pp']-b['drop_error_pp']);maxdrop=max(maxdrop,difference);assert difference<1e-10
            for split in cc['splits']:
                with np.load(folder/split['name']/f'candidate_{i}_predictions.npz') as a,np.load(folder/split['name']/f'candidate_{i+6}_predictions.npz') as b:
                    for key in split['validation']:
                        delta=b[key+'__100__prediction']-a[key+'__100__prediction'];variation=float(np.ptp(delta));maxvariation=max(maxvariation,variation);assert variation<1e-12;pairs+=1
    def stats(group):
        return dict(records=len(group),initial_abs_pp=float(np.mean([abs(r['initial_signed_pp']) for r in group])),trend_abs_pp=float(np.mean([abs(r['trend_signed_pp']) for r in group])),endpoint_pp=float(np.mean([r['endpoint_pp'] for r in group])))
    result=dict(passed=True,paired_curves=pairs,max_same_weight_drop_error_difference_pp=maxdrop,max_vertical_offset_variation_soh=maxvariation,selected=stats(rows),local_protocol=stats(local),family_fallback=stats(fallback),original_validation_loaded=False,outer_loaded=False,independent_batteries=False,interpretation='Same-weight anchoring changes only absolute level, not degradation shape. Training parent pools overlap; descriptive diagnostics only, no oracle query correction.',verification_sha256=sha(proof),source_sha256=sha(__file__),goal_complete=False)
    save_json(root/'shape_error_diagnostic.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
