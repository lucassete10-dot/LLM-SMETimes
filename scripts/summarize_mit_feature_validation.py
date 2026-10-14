"""Descriptive decomposition of frozen inner mean plus P90 score; no selection changes."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.prepare_same_type_forecast import save_json,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);root=ap.parse_args().run
    cfg=json.loads((root/'config.json').read_text());sel=json.loads((root/'selection.json').read_text());records=[];inputs={}
    for i in range(len(cfg['candidates'])):
        f=root/f'candidate_{i}'/'validation_metrics.json';rows=json.loads(f.read_text());assert [r['cell_id'] for r in rows]==cfg['plan']['validation']
        values=np.array([(r['mape_percent']+r['late_mape']+r['endpoint_pp']+r['drop_error_pp'])/4 for r in rows])
        mean=float(values.mean());p90=float(np.quantile(values,.9));score=mean+.25*p90
        np.testing.assert_allclose(score,sel['candidates'][i]['score'],atol=1e-12,rtol=0)
        records.append(dict(candidate=i,parameters=cfg['candidates'][i],cell_count=len(rows),
            mean_cell_score=mean,p90_cell_score=p90,selection_score=score,
            joint_pass_cells=sum(r['mape_percent']<=6 and r['late_mape']<=8 and r['endpoint_pp']<=8 and r['drop_error_pp']<=5 for r in rows)))
        inputs[str(f)]=sha(f)
    save_json(root/'validation_score_decomposition.json',dict(rows=records,source_sha256=sha(__file__),input_sha256=inputs,
        original_selection_unchanged=True,outer_loaded=False,goal_complete=False))
    print(json.dumps(records,indent=2),flush=True)

if __name__=='__main__':main()
