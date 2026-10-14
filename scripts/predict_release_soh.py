"""Predict without labels. JSON request uses physical specs, not dataset identity."""
import argparse
import json
from pathlib import Path
from battery_smetimes.external_soh import load_context,read_signals,route,forecast
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--signals',type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--request',type=Path);mode.add_argument('--public-replay-cell')
    p.add_argument('--end-cycle',type=int,required=True);p.add_argument('--step',type=int,default=1);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    context=load_context(args.package);cycle,raw=read_signals(args.signals)
    request=json.loads(args.request.read_text(encoding='utf-8')) if args.request else None
    cell,provenance=route(context,cycle,raw,request,args.public_replay_cell)
    result=forecast(context,cell,provenance,args.end_cycle,args.step)
    files=[args.signals,Path(__file__),Path('battery_smetimes/external_soh.py')]
    if args.request:files.append(args.request)
    result['source_sha256']={str(file):sha(file) for file in files}
    result['request']=request if request is not None else dict(public_replay_cell=args.public_replay_cell)
    args.output.parent.mkdir(parents=True,exist_ok=True);save_json(args.output,result)
    print('OUTPUT='+str(args.output.resolve()))


if __name__=='__main__':main()
