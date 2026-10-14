"""Predict future SOH from observed signals and known metadata, without labels."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from battery_smetimes.stm32_forecast import predict
from scripts.prepare_same_type_forecast import save_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--signals',type=Path,required=True)
    parser.add_argument('--metadata',type=Path,required=True)
    parser.add_argument('--end-cycle',type=int,required=True)
    parser.add_argument('--step',type=int,default=1)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--float',action='store_true',dest='float_model')
    args=parser.parse_args()
    if args.end_cycle<=100 or args.step<=0: raise ValueError('query cycles must be >100 with positive step')
    bundle=joblib.load(args.model)
    metadata=json.loads(args.metadata.read_text(encoding='utf-8'))
    with np.load(args.signals,allow_pickle=False) as value:
        cell={'cycle':value['cycle'],'raw':value['raw_features'],'metadata':metadata['metadata'],
              'physical_family':metadata['physical_family'],
              'reference_established_source_cycle':metadata['reference_established_source_cycle']}
    if np.any(cell['cycle']>100): raise ValueError('provide only observations available through source cycle100')
    future=np.arange(101,args.end_cycle+1,args.step)
    prediction=predict(dict(bundle,quantized=not args.float_model),cell,100,future)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    save_json(args.output,{'origin_source_cycle':100,'physical_family':cell['physical_family'],
        'mode':'float' if args.float_model else 'int8','future_source_cycles':future.tolist(),
        'predicted_soh':prediction.astype(float).tolist(),
        'input_contract':'observed V/I derived features only; no capacity/SOH/future signals/total life'})
    print(f'Saved {len(future)} future predictions to {args.output}')


if __name__=='__main__': main()
