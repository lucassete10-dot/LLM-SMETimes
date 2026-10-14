"""Lock reference selections against PREVIOUSLY recorded outer-config SHA.

Only historical configuration/selection routing is read; no test arrays or
test metrics participate. This companion records provenance, not new choices.
"""
import argparse,json,shutil
from datetime import datetime,timezone
from pathlib import Path
from scripts.train_curve_fidelity import AUDIT
from scripts.prepare_same_type_forecast import save_json,sha

ROUTES={'MIT':Path('experiments/curve_fidelity_outer/20261005T030425_342025Z/config.json'),
    'XJTU':Path('experiments/curve_fidelity_outer/20261005T023843_153503Z/config.json'),
    'NASA':Path('experiments/curve_fidelity_outer/20261005T032919_625507Z/config.json'),
    'Zhu2022_NCA':Path('experiments/nca_curve_fidelity_outer/20261005T033153_613655Z/config.json')}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    run=parser.parse_args().run;cfg=json.loads((run/'config.json').read_text());assert not cfg['smoke']
    refs=[]
    for plan in cfg['plans']:
        ref=cfg['frozen_references'][plan['name']];source=Path(ref['run'])
        assert sha(source/'config.json')==ref['config_sha256']
        route_file=ROUTES[plan['dataset']];route=json.loads(route_file.read_text())
        if ref['layout']=='core':
            item=next(r for r in route['validation_sources'] if Path(r['run'])==source)
            expected=item['selection_sha256'];assert item['config_sha256']==ref['config_sha256']
        else:
            assert Path(route['validation_source'])==source
            assert route['validation_config_sha256']==ref['config_sha256']
            expected=route['selection_sha256'][plan['name']]
        for seed in cfg['seeds']:
            p=source/str(seed)/'selection.json' if ref['layout']=='core' else source/plan['name']/str(seed)/'selection.json'
            assert sha(p)==expected[str(seed)]
            selected=json.loads(p.read_text());assert selected['selected']==min(selected['candidates'],key=lambda r:r['selection_score'])
        refs.append(dict(fold=plan['name'],run=str(source),layout=ref['layout'],
            reference_config_sha256=ref['config_sha256'],selection_sha256=expected,
            prior_route_config=str(route_file),prior_route_sha256=sha(route_file)))
    snapshot=run/'input_provenance';snapshot.mkdir(exist_ok=True)
    auditcfg=json.loads((AUDIT/'config.json').read_text())
    registry=AUDIT/'source_snapshot/battery_physical_registry.json'
    assert sha(registry)==auditcfg['registry_sha256']
    integrity=Path(auditcfg['config']['integrity_audit']);assert sha(integrity)==auditcfg['integrity_sha256']
    sources=[AUDIT/'config.json',AUDIT/'cell_registry.json',registry,integrity]
    for i,p in enumerate(sources):shutil.copy2(p,snapshot/(str(i)+'_'+p.name))
    save_json(run/'reference_lock.json',dict(passed=True,references=refs,source_sha256=sha(__file__),
        core_input_sources=[dict(path=str(p),sha256=sha(p)) for p in sources],
        primary_policy=cfg['primary_outer_policy'],locked_at=datetime.now(timezone.utc).isoformat(),
        note='Companion after launch checks hashes already recorded in previous stage; no new selection or outer metric access.'))
    print('LOCKED='+str(len(refs))+' folds')


if __name__=='__main__':main()
