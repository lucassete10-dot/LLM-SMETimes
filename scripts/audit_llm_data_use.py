"""Index existing prediction archives and declared training membership.

Absence from these artifacts is not proof that data was never viewed. This
ledger deliberately does not grant an independent-confirmation designation.
"""
import json,subprocess,zipfile
from pathlib import Path
from scripts.audit_llm_curve_repair import load_frozen,fingerprint
from scripts.prepare_same_type_forecast import save_json


def main():
    root=Path(json.loads(Path('experiments/llm_curve_repair/latest_P0.json').read_text())['folder'])
    cfg=json.loads((root/'config.json').read_text());manifest,cells,_,_,_=load_frozen(cfg)
    datasets={d:{k for k,c in cells.items() if c['metadata']['dataset']==d} for d in ['MIT','XJTU']}
    indexes={
        'HUST':Path('data/interim/hust_voltage_intervals/20261008T074246_799565Z/index.json'),
        'UConn_ISU_ILCC':Path('data/interim/uconn_causal_cycles/20261008T073156_095768Z/index.json'),
        'Zhu2022_NCA':Path('data/interim/nca_curve_cells/20261005T031838_298841Z/index.json')}
    for domain,path in indexes.items():
        collection=json.loads(path.read_text())['cells']
        datasets[domain]=set(collection) if isinstance(collection,dict) else {r['cell_id'] for r in collection}
    ids=set.union(*datasets.values());events={cid:{'first_prediction_archive':None,'archive_count':0,'declared_training_config':None} for cid in ids}
    paths=subprocess.run(['rg','--files','experiments','-g','*.npz','-g','config.json'],capture_output=True,text=True,check=True).stdout.splitlines()
    failures=[];counts={'prediction_archives_scanned':0,'config_files_scanned':0}
    for name in paths:
        p=Path(name)
        try:
            if p.suffix=='.npz':
                with zipfile.ZipFile(p) as archive:
                    known={item[:-4].split('__')[0] for item in archive.namelist() if item.endswith('__prediction.npy')}
                    truths={item[:-4].split('__')[0] for item in archive.namelist() if item.endswith('__truth.npy')}
                for cid in known & truths & ids:
                    events[cid]['archive_count']+=1
                    if events[cid]['first_prediction_archive'] is None:events[cid]['first_prediction_archive']=str(p)
                counts['prediction_archives_scanned']+=1
            else:
                data=json.loads(p.read_text(encoding='utf-8'));found=set()
                def walk(item):
                    if isinstance(item,dict):
                        for key,value in item.items():
                            if key in ('train','train_ids','used_train_ids','train_cells','training') and isinstance(value,list):
                                found.update(v for v in value if isinstance(v,str) and v in ids)
                            walk(value)
                    elif isinstance(item,list):
                        for value in item:walk(value)
                walk(data)
                for cid in found:
                    if events[cid]['declared_training_config'] is None:events[cid]['declared_training_config']=str(p)
                counts['config_files_scanned']+=1
        except (ValueError,OSError,zipfile.BadZipFile) as error:failures.append({'path':name,'error':str(error)})
    rows={}
    for domain,keys in datasets.items():
        available={cid:events[cid] for cid in sorted(keys)}
        candidates=[cid for cid,row in available.items() if not row['first_prediction_archive'] and not row['declared_training_config']]
        rows[domain]={'indexed_cells':len(keys),'with_saved_predictions':sum(bool(r['first_prediction_archive']) for r in available.values()),
            'with_declared_training':sum(bool(r['declared_training_config']) for r in available.values()),
            'no_indexed_prior_use_candidates':candidates,'by_cell':available,
            'independent_confirmation_proven':False}
    report={'scope':'Artifact-based historical-use audit only; no new heldout prediction/label inspection',
        'scan_counts':counts,'failures':failures,'datasets':rows,
        'index_sha256':{k:fingerprint(v) for k,v in indexes.items()},'source_sha256':fingerprint(Path(__file__)),
        'limitations':'Metadata cannot certify unseen data; unindexed joblib checkpoints, notebooks, external analyses and prior viewing remain unaudited. No available cell is silently renamed independent.',
        'goal_complete':False}
    save_json(root/'data_use_ledger.json',report)
    print(json.dumps({k:{a:b for a,b in v.items() if a!='by_cell'} for k,v in rows.items()},indent=2),flush=True)


if __name__=='__main__':main()
