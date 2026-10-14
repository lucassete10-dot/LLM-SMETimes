"""Report every failed held-out cell; never select models from outer outcomes."""
import json
from pathlib import Path
from collections import Counter
from scripts.prepare_same_type_forecast import sha,save_json

ROOT=Path('experiments/mit_feature_blend_outer/20261006T144031_590137Z')


def main():
    proof=json.loads((ROOT/'independent_verification.json').read_text());assert proof['passed']
    rows=json.loads((ROOT/'by_cell.json').read_text());assert len(rows)==134
    failures=[];counts=Counter()
    for row in rows:
        reasons=[]
        for key,limit in [('mape_percent',6),('late_mape',8),('endpoint_pp',8),('drop_error_pp',5)]:
            if row[key]>limit:reasons.append(key);counts[key]+=1
        assert row['joint_pass']==(not reasons)
        if reasons:failures.append(dict(**row,reasons=reasons))
    assert len(failures)==46 and sum(r['joint_pass'] for r in rows)==88
    result=dict(count=134,joint_pass_cells=88,failed_cells=46,criterion_failure_counts=dict(counts),
        overlapping_failure_reasons=True,failures=failures,
        original_by_cell_sha256=sha(ROOT/'by_cell.json'),outer_proof_sha256=sha(ROOT/'independent_verification.json'),
        source_sha256=sha(__file__),scope='complete reporting only; not training or model selection',goal_complete=False)
    save_json(ROOT/'complete_failure_report.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='failures'}),flush=True)


if __name__=='__main__':main()
