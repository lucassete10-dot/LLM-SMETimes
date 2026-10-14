"""Clarify component versus legacy-bundle units in the outer refit counter."""
import argparse,json
from pathlib import Path
import joblib
from scripts.prepare_same_type_forecast import save_json,sha

def count(bundle):
    assert 'model' in bundle and 'local' in bundle
    return 1+sum(count(child) for child in bundle['local'].values())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);root=ap.parse_args().run
    proof=json.loads((root/'independent_verification.json').read_text());assert proof['passed']
    cfg=json.loads((root/'config.json').read_text());rows=[]
    for p in cfg['plans']:
        for s in cfg['seeds']:
            file=root/p['name']/str(s)/'model.joblib';b=joblib.load(file)
            rows.append(dict(fold=p['name'],seed=s,mode=b['mode'],estimator_components=count(b['inner']),model_sha256=sha(file)))
    new=sum(r['estimator_components'] for r in rows if r['mode']=='segment');old=sum(r['estimator_components'] for r in rows if r['mode']!='segment');oldbundles=sum(r['mode']!='segment' for r in rows)
    assert proof['component_refits']==new+oldbundles
    save_json(root/'component_refit_units.json',dict(rows=rows,new_individual_component_refits=new,legacy_bundle_refits=oldbundles,legacy_estimator_components_refit=old,total_estimator_components=new+old,original_proof_counter=proof['component_refits'],meaning='Original counter counted each old bundle as one; fresh legacy fitting and exact state comparison covered all its local estimators. No model or score changed.',source_sha256=sha(__file__)))
    print('Estimator components '+str(new+old)+'; original counter '+str(proof['component_refits'])+' mixes components and old bundles',flush=True)

if __name__=='__main__':main()
