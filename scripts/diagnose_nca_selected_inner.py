"""Diagnose all locked NCA inner winners, not observed outer failed cells."""
import json,shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from battery_smetimes.tail_selection import choose
from scripts.train_tail_boost import load_cells
from scripts.prepare_same_type_forecast import sha,save_json
from scripts.diagnose_xjtu_segment_inner_shape import decomposition

PARENT=Path('experiments/tail_boost/20261005T034431_681399Z')


def main():
    cfg=json.loads((PARENT/'config.json').read_text())
    plans=[p for p in cfg['plans'] if p['dataset']=='Zhu2022_NCA']
    assert len(plans)==5 and len(cfg['seeds'])==5
    root=Path('experiments/nca_selected_inner_diagnostic')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True);print('RUN='+str(root.resolve()),flush=True)
    snap=root/'source_snapshot';snap.mkdir()
    files=[Path(__file__),Path('battery_smetimes/tail_selection.py'),Path('scripts/diagnose_xjtu_segment_inner_shape.py')]
    for f in files:shutil.copy2(f,snap/f.name)
    rows=[];dependencies={};choices=[]
    for plan in plans:
        cells=load_cells(plan);assert not set(cells)&set(plan['test'])
        for seed in cfg['seeds']:
            choice=choose(PARENT,cfg,plan,seed,cells)['selected'];choices.append(dict(fold=plan['name'],seed=seed,**choice))
            sf=Path(choice['selection_file']);file=sf.parent/f"candidate_{choice['candidate']}"/'validation_predictions.npz'
            dependencies[str(sf)]=sha(sf);dependencies[str(file)]=sha(file)
            with np.load(file) as a:
                assert set(a.files)=={k+'__100__'+f for k in plan['validation'] for f in ('cycle','truth','prediction')}
                for k in plan['validation']:
                    c,y,p=[a[k+'__100__'+f] for f in ('cycle','truth','prediction')]
                    np.testing.assert_array_equal(c,cells[k]['cycle']);np.testing.assert_array_equal(y,cells[k]['truth'])
                    rows.append(dict(fold=plan['name'],seed=seed,cell_id=k,mode=choice['mode'],**decomposition(c,y,p)))
            print('READ '+plan['name']+' seed '+str(seed),flush=True)
    summary=dict(records=len(rows),unique_validation_cells=len({r['cell_id'] for r in rows}),
        mean_start_abs_pp=float(np.mean([abs(r['start_signed_pp']) for r in rows])),
        mean_endpoint_abs_pp=float(np.mean([abs(r['endpoint_signed_pp']) for r in rows])),
        mean_drop_error_pp=float(np.mean([r['drop_abs_pp'] for r in rows])),
        mean_signed_change_pp=float(np.mean([r['change_signed_pp'] for r in rows])),
        underdecline_fraction=float(np.mean([r['underdecline'] for r in rows])),
        severe_underdecline_fraction=float(np.mean([r['severe_underdecline'] for r in rows])),
        late_underdecline_fraction=float(np.mean([r['late_underdecline'] for r in rows])),
        drop_dominates_start_fraction=float(np.mean([r['drop_abs_pp']>abs(r['start_signed_pp']) for r in rows])),
        interpretation='Overlapping inner records; not outer accuracy or causal attribution.',outer_loaded=False,new_fits=0,goal_complete=False)
    save_json(root/'config.json',dict(parent=str(PARENT.resolve()),parent_config_sha256=sha(PARENT/'config.json'),
        source_sha256={str(f):sha(f) for f in files},input_sha256=dependencies,plans=plans,seeds=cfg['seeds'],outer_loaded=False,new_fits=0))
    save_json(root/'choices.json',choices);save_json(root/'rows.json',rows);save_json(root/'summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
