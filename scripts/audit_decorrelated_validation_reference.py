"""Compare registered new validation scores to current frozen primary, not weak control."""
import argparse,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.tail_selection import choose
from scripts.train_tail_boost import load_cells
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text())
    assert cfg['fivefold'] and json.loads((root/'progress.json').read_text())['training_complete']
    routes={}
    for entry in cfg['provenance']:
        path=Path(entry['file']);assert sha(path)==entry['sha256'];c=json.loads(path.read_text())
        routes[c['plans'][0]['dataset']]=(path.parent,c)
    reports=[];new=json.loads((root/'summary.json').read_text())
    for plan in cfg['plans']:
        source,scfg=routes[plan['dataset']];cells=load_cells(plan)
        assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(plan['test'])
        choice=choose(source,scfg,plan,cfg['seed'],cells)
        candidate=next(r for r in new if r['fold']==plan['name'] and r['candidate']==1)
        reports.append(dict(dataset=plan['dataset'],fold=plan['name'],seed=cfg['seed'],new_score=candidate['score'],primary_score=choice['selected']['score'],new_beats_primary=bool(candidate['score']<choice['selected']['score']),primary=choice['selected']))
        print(plan['name']+' new='+str(candidate['score'])+' primary='+str(choice['selected']['score']),flush=True)
    summary={d:dict(new_better_folds=sum(r['new_beats_primary'] for r in reports if r['dataset']==d),folds=sum(r['dataset']==d for r in reports),mean_score_delta=float(np.mean([r['new_score']-r['primary_score'] for r in reports if r['dataset']==d]))) for d in routes}
    save_json(root/'current_primary_validation_comparison.json',dict(reports=reports,summary=summary,used_only_train_validation=True,outer_loaded=False,goal_complete=False,source_sha256=sha(__file__)))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
