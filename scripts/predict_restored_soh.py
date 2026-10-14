"""Predict with the restored MIT ensemble from an existing feature prefix."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from battery_smetimes.physical_types import match_external_specifications
from scripts.restore_soh_curve_comparison import tree_input
from scripts.diagnose_soh_curve_small import save


def predict(release_path, signal_path, metadata_path, end_cycle):
    release=json.loads(Path(release_path).read_text(encoding='utf-8'))
    metadata=json.loads(Path(metadata_path).read_text(encoding='utf-8'))
    if release['schema']!='soh_curve_restoration_tree_ensemble_v1':raise ValueError('unsupported release schema')
    for source,digest in release['input_sources_sha256'].items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=digest:raise ValueError('input implementation differs from saved release')
    match=match_external_specifications(metadata.get('specifications',{}),registry=release['physical_registry'])
    if match['status']!='matched' or match['physical_family']!=release['physical_family']:
        raise ValueError('battery specifications do not match this trained physical family')
    if metadata.get('dataset')!='MIT':raise ValueError('MIT feature schema required')
    if metadata.get('cell_id') in release['train_ids']:raise ValueError('this cell was used for fitting; not an unseen-cell forecast')
    origin=release['origin']
    if end_cycle<=origin:raise ValueError('end cycle must be after origin')
    with np.load(signal_path,allow_pickle=False) as data:
        if set(data.files)!={'cycle','raw_features'}:raise ValueError('only cycle/raw_features are allowed as model input')
        cycle,raw=data['cycle'],data['raw_features']
    if np.any(cycle>origin):raise ValueError('provide only the first100 source-cycle signal prefix')
    cell=dict(metadata=metadata,cycle=cycle,raw=raw,reference_established_source_cycle=0)
    future=np.arange(origin+1,end_cycle+1,dtype=float)
    values=[]
    for record in release['models']:
        path=Path(record['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('saved model hash changed')
        bundle=joblib.load(path)
        if bundle['physical_family']!=release['physical_family'] or bundle['train_ids']!=release['train_ids']:raise ValueError('model/release identity mismatch')
        x=tree_input(cell,future,bundle['types'])
        values.append(bundle['model'].predict(bundle['imputer'].transform(x)))
    values=np.asarray(values)
    if not np.isfinite(values).all():raise ValueError('nonfinite predictions')
    return dict(physical_family=release['physical_family'],origin_source_cycle=origin,
                future_source_cycles=future.tolist(),predicted_soh=values.mean(0).tolist(),
                seed_standard_deviation=values.std(0,ddof=1).tolist(),
                soh_definition='comparable discharge capacity / initial comparable reference capacity',
                scope='development model; same known operating strategy; seed SD is not calibrated uncertainty',
                input_schema='51 MIT features from the existing extractor, not arbitrary raw V/I samples')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--signals',type=Path,required=True);parser.add_argument('--metadata',type=Path,required=True)
    parser.add_argument('--end-cycle',type=int,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('output already exists')
    result=predict(args.release,args.signals,args.metadata,args.end_cycle)
    args.output.parent.mkdir(parents=True,exist_ok=True);save(args.output,result);print(args.output)


if __name__=='__main__':main()
