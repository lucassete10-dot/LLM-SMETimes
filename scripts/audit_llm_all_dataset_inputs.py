"""All-fold CPU input isolation audit; no extra training or TEST selection."""
import argparse,copy,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from battery_smetimes.research_sequences import HistoryPreprocessor
from battery_smetimes.research_llm import statistics_cards
from scripts.audit_llm_curve_repair import load_frozen,fingerprint
from scripts.prepare_same_type_forecast import save_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--queue',type=Path,required=True);args=ap.parse_args()
    queue=json.loads((args.queue/'queue.json').read_text(encoding='utf-8'));plans={}
    for task in queue['tasks']:plans.setdefault(task['fold'],task)
    records=[];failures=[];folds=[]
    with threadpool_limits(limits=2):
        for fold,task in plans.items():
            cfg=json.loads(Path(task['settings']).read_text(encoding='utf-8'));_,cells,plan,_,_=load_frozen(cfg)
            processor=HistoryPreprocessor().fit(cells,plan['train'],[cfg['origin']],history_slots=cfg['history_slots'])
            if processor.train_ids!=plan['train']:raise ValueError('preprocessor membership differs from TRAIN')
            for role in ('train','validation','test'):
                for cid in plan[role]:
                    try:
                        cell=cells[cid];origin=cfg['origin'];visible=cell['cycle']<=origin
                        before=processor.model_input(cell,origin);cards=statistics_cards(cell,origin,history_slots=cfg['history_slots'])
                        poisoned=copy.copy(cell)
                        for key in ('soh','label_cycle','unfiltered_soh','capacity','capacity_ah','reference_capacity_ah','eol','lifetime','cycle_life'):poisoned.pop(key,None)
                        poisoned['metadata']={**cell['metadata'],'capacity':float('inf'),'soh':float('inf'),'lifetime':float('inf')}
                        poisoned['cycle']=np.r_[cell['cycle'][visible],origin+100,origin+200]
                        # Intentional float64 promotion tests canonicalized cards
                        # and numeric statistics alongside visible-prefix masks.
                        poisoned['raw']=np.r_[cell['raw'][visible],np.full((2,cell['raw'].shape[1]),np.inf)]
                        after=processor.model_input(poisoned,origin)
                        for key in ('history','statistics','conditions'):np.testing.assert_array_equal(before[key],after[key])
                        if before['length']!=after['length'] or cards!=statistics_cards(poisoned,origin,history_slots=cfg['history_slots']):raise ValueError('visible input/card changed')
                        schema=cell['metadata'].get('signal_schema')
                        if schema:
                            for channel,columns in schema['text_channel_columns'].items():
                                if not columns and any(channel+' unavailable.' not in text for text in cards):raise ValueError('missing channel represented as measured')
                        records.append({'fold':fold,'dataset':task['dataset'],'cell_id':cid,'role':role,'passed':True,
                            'visible_observations':before['length'],'raw_width':cell['raw'].shape[1],
                            'last_visible_source_cycle':float(cell['cycle'][visible][-1])})
                    except Exception as error:failures.append({'fold':fold,'dataset':task['dataset'],'cell_id':cid,'role':role,'error':str(error)})
            folds.append({'fold':fold,'dataset':task['dataset'],'requested':sum(len(plan[r]) for r in ('train','validation','test')),'TRAIN_fit_ids':processor.train_ids,
                'settings_sha256':fingerprint(Path(task['settings']))})
            print(json.dumps({'fold':fold,'checked':sum(r['fold']==fold for r in records),'failures':sum(r['fold']==fold for r in failures)}),flush=True)
    requested=sum(f['requested'] for f in folds)
    output={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'scope':'All frozen folds; fixed TRAIN-only preprocessors; numeric input and statistics-card equality under label deletion and future contamination. No model forecasts or algorithm selection.',
        'requested_fold_cell_instances':requested,'checked_fold_cell_instances':len(records),'distinct_source_cells':len({r['cell_id'] for r in records}),
        'folds':folds,'per_cell':records,'failures':failures,'passed':not failures and len(records)==requested,
        'source_sha256':{str(p):fingerprint(p) for p in (Path(__file__),Path('battery_smetimes/research_sequences.py'),Path('battery_smetimes/research_llm.py'),Path('battery_smetimes/llm_external_schema.py'))},
        'not_prediction_isolation_or_accuracy_evidence':True,'goal_complete':False}
    save_json(args.queue/'all_dataset_input_isolation_audit.json',output)
    print(json.dumps({k:output[k] for k in ('requested_fold_cell_instances','checked_fold_cell_instances','distinct_source_cells','passed')},indent=2),flush=True)
    if not output['passed']:raise ValueError('input contract audit failed; preserve all failures')


if __name__=='__main__':main()
