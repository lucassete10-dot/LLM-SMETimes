"""Training-side comparison to frozen current primary, never outer artifacts."""
import argparse,json
from pathlib import Path
from threadpoolctl import threadpool_limits
from battery_smetimes.tail_selection import choose
from scripts.train_tail_boost import load_cells
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text())
    assert json.loads((root/'independent_verification.json').read_text())['passed']
    rows=json.loads((root/'summary.json').read_text());reports=[]
    routes={'MIT':'20261005T034441_870182Z','XJTU':'20261005T034452_066902Z','Zhu2022_NCA':'20261005T034431_681399Z'}
    for plan in cfg['plans']:
        new=min([r for r in rows if r['fold']==plan['name'] and r['candidate']!=0],key=lambda r:r['score'])
        if plan['dataset']=='NASA':
            control=next(r for r in rows if r['fold']==plan['name'] and r['candidate']==0)
            reports.append(dict(dataset='NASA',fold=plan['name'],new=new,weak_control_score=control['score'],strong_primary_replayed=False,reason='Current tree-control comparison is insufficient to claim strong NASA Ridge superiority.'))
            continue
        source=Path('experiments/tail_boost')/routes[plan['dataset']];scfg=json.loads((source/'config.json').read_text())
        source_plan=next(p for p in scfg['plans'] if p['name']==plan['name'])
        for role in ('train','validation','test'):assert source_plan[role]==plan[role]
        cells=load_cells(plan);assert set(cells)==set(plan['train']+plan['validation']) and not set(cells)&set(plan['test'])
        primary=choose(source,scfg,plan,cfg['seed'],cells)
        report=dict(dataset=plan['dataset'],fold=plan['name'],new=new,primary=primary,new_beats_primary=bool(new['score']<primary['selected']['score']),primary_config_sha256=sha(source/'config.json'))
        reports.append(report);print(plan['name']+' new='+str(new['score'])+' primary='+str(primary['selected']['score']),flush=True)
    save_json(root/'current_primary_validation_comparison.json',dict(reports=reports,used_only_train_validation=True,outer_loaded=False,goal_complete=False,source_sha256=sha(__file__)))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
