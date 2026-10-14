"""Full independent curve replay plus five-fold expansion rule verification."""
import json,sys
from pathlib import Path
import numpy as np
from scripts.verify_neighbor_curve_smoke import main as verify_curves
from scripts.prepare_same_type_forecast import sha,save_json

def decision(results):
    wins=sum(r['selected']['score']<r['control']['score'] for r in results)
    means={role:{f:float(np.mean([r[role][f] for r in results])) for f in ('score','endpoint_pp','drop_error_pp','mape_percent')} for role in ('selected','control')}
    expand=wins>=4 and means['selected']['score']<means['control']['score'] and all(means['selected'][f]<=means['control'][f] for f in ('endpoint_pp','drop_error_pp'))
    return wins,means,expand

def main():
    verify_curves();root=Path(sys.argv[sys.argv.index('--run')+1]);cfg=json.loads((root/'config.json').read_text());assert len(cfg['plans'])==5 and len({p['name'] for p in cfg['plans']})==5
    records=[json.loads((root/p['name']/'selection.json').read_text()) for p in cfg['plans']]
    assert records==json.loads((root/'fold_results.json').read_text())
    wins,means,expand=decision(records);saved=json.loads((root/'screening_decision.json').read_text());assert saved['fold_score_wins']==wins and saved['expand_formal']==expand
    for role in means:
        for field,v in means[role].items():np.testing.assert_allclose(v,saved['means'][role][field],rtol=0,atol=1e-12)
    save_json(root/'independent_screening_verification.json',dict(passed=True,fold_score_wins=wins,means=means,expand_formal=expand,outer_loaded=False,source_sha256=sha(__file__),goal_complete=False));print(json.dumps(dict(wins=wins,expand_formal=expand)),flush=True)

if __name__=='__main__':main()
