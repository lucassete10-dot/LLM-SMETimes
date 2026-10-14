"""Isolated development experiment; never loads the original fold's test cells."""
import copy
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from battery_smetimes.research_sequences import HistoryPreprocessor, SmallForecaster, training_contexts, pack_contexts

AUDIT = Path('experiments/soh_research_v5/audits/20260930T074750_792569Z')


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


class CurveModel(nn.Module):
    def __init__(self, processor, spline=False, time_only=False):
        super().__init__()
        self.base = SmallForecaster('statistics_mlp', 2*processor.raw_width+1, processor.stats_width, 64, .1)
        self.spline, self.time_only = spline, time_only
        self.coefficients = nn.Linear(73, 10) if spline else None
        self.register_buffer('knots', torch.tensor([0., .25, .5, .75, 1., 1.5, 2., 3.]))
        self.register_buffer('query_mean', torch.tensor(processor.query_scale.scaler.mean_, dtype=torch.float32))
        self.register_buffer('query_scale', torch.tensor(processor.query_scale.scaler.scale_, dtype=torch.float32))

    def forward(self, batch):
        stats, conditions = batch['statistics'], batch['conditions']
        if self.time_only:
            stats = stats.clone()
            stats[:, :-5] = 0
            conditions = conditions.clone()
            conditions[:, :-5] = 0
        encoded = self.base.encode(batch['history'], batch['length'], stats)
        if not self.spline:
            return self.base.decode(encoded, conditions, batch['query'])
        coef = self.coefficients(torch.cat((encoded, conditions), -1))
        # Requested absolute cycle only, not true lifespan or future observations.
        t = (batch['query'][:,:,0]*self.query_scale[0]+self.query_mean[0])/10
        basis = F.relu(t[:,:,None]-self.knots)
        return (1+.1*coef[:,0,None] - .05*F.softplus(coef[:,1,None])*t
                - .05*(basis*F.softplus(coef[:,None,2:])).sum(-1))


def metrics(y, p):
    e = p-y
    total = float(np.sum((y-y.mean())**2))
    return dict(mape=float(np.mean(abs(e)/y)*100), mae_pp=float(abs(e).mean()*100),
                rmse_pp=float(np.sqrt(np.mean(e**2))*100), r2=1-float(np.sum(e**2))/total if total else None,
                late_mape=float(np.mean(abs(e[len(e)//2:])/y[len(e)//2:])*100),
                endpoint_error_pp=float(abs(e[-1])*100))


def evaluate(model, contexts):
    model.eval()
    rows, arrays = [], {}
    with torch.inference_mode():
        for context in contexts:
            pred = model(pack_contexts([context], 'cpu'))[0].numpy()
            valid = context['valid'].astype(bool)
            y, p = context['truth'][valid], pred[valid]
            rows.append({'cell_id':context['cell_id'], **metrics(y,p)})
            arrays[context['cell_id']+'__truth'] = y
            arrays[context['cell_id']+'__prediction'] = p
    return rows, arrays


def mean_score(rows):
    return float(np.mean([(r['mape']+r['late_mape'])/2 for r in rows]))


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    out = Path('experiments/soh_curve_diagnostics')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True)
    split = next(p for p in json.loads((AUDIT/'splits.json').read_text())['plans'] if p['name']=='MIT_fold3')
    rng = np.random.default_rng(20261001)
    chosen = {}
    for batch in ('b1','b2','b3'):
        keys = rng.permutation(sorted(k for k in split['train'] if k.startswith(batch))).tolist()
        chosen[batch] = keys[:10]
    train = [k for keys in chosen.values() for k in keys[:8]]
    inner = [k for keys in chosen.values() for k in keys[8:]]
    validation = list(split['validation'])
    assert not (set(train)&set(inner) or set(train+inner)&set(validation) or set(train+inner+validation)&set(split['test']))
    config = dict(scope='development-only MIT small-model proxy; not a new blind test or proof of LLM cause',
                  train=train, inner_validation=inner, diagnostic_validation=validation,
                  excluded_original_test=split['test'], origin=100, seeds=list(range(20261001,20261006)),
                  epochs=160, validation_every=5, lr=.001, weight_decay=.01, target_cap=96,
                  selection='inner cell-macro (all MAPE + late-half MAPE)/2; identical for all variants',
                  variants=['original_head','original_head_tail','curve_head','curve_head_tail','time_conditions_only'],
                  tail_loss='linearly weighted 1 to 4 across training future observations; no test lifespan input',
                  curve_head='monotone accelerating hinge basis, fixed absolute-cycle knots; MIT diagnostic only',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save(out/'config.json',config)
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    registry = json.loads((AUDIT/'cell_registry.json').read_text())['cells']
    issues = json.loads(Path('data/interim/same_type_forecast/mit_discharge_integrity_20260929T060206_688121Z.json').read_text())['issues']
    cells={}
    for key in train+inner+validation:
        record=registry[key]
        path=Path(record['prepared_file'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['prepared_sha256']
        with np.load(path) as data:
            cell={**record,'raw':data['raw_features'].copy(),'cycle':data['cycle'].copy(),
                  'soh':data['soh'].copy(),'label_cycle':data['label_cycle'].copy()}
        rejected=[r['cycle'] for r in issues if r['cell_id']==key]
        cell['soh'][np.isin(cell['label_cycle'],rejected)]=np.nan
        keep=~np.isin(cell['cycle'],rejected)
        cell['cycle'],cell['raw']=cell['cycle'][keep],cell['raw'][keep]
        cells[key]=cell
    assert len({c['physical_family'] for c in cells.values()})==1
    save(out/'data_provenance.json',{k:{'sha256':registry[k]['prepared_sha256'],'physical_family':registry[k]['physical_family']} for k in cells})
    processor=HistoryPreprocessor().fit(cells,train,[100])
    joblib.dump(processor,out/'preprocessor.joblib')
    contexts=training_contexts(cells,train,processor)
    inner_contexts=training_contexts(cells,inner,processor)
    # Full observed future grid for diagnostic scores, no downsampling.
    fullcap=max(len(c['label_cycle']) for c in cells.values())
    validation_contexts=training_contexts(cells,validation,processor,cap=fullcap)
    train_full=training_contexts(cells,train,processor,cap=fullcap)
    # Contract: changing future labels must not affect any model input.
    probe=copy.deepcopy(cells[validation[0]])
    before=processor.model_input(probe,100)
    probe['soh'][:]=42; probe['label_cycle'][:]=999999
    after=processor.model_input(probe,100)
    assert all(np.array_equal(before[k],after[k]) for k in before)
    save(out/'checks.json',{'disjoint_cells':True,'train_only_scaling':processor.train_ids==train,
                           'future_label_input_invariance':True,'original_test_files_loaded':0})
    summaries=[]
    for variant in config['variants']:
        for seed in config['seeds']:
            started=time.perf_counter()
            torch.manual_seed(seed)
            model=CurveModel(processor,spline=variant.startswith('curve'),time_only=variant=='time_conditions_only')
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
            best=float('inf'); history=[]
            batch=pack_contexts(contexts,'cpu')
            for epoch in range(1,config['epochs']+1):
                model.train(); p=model(batch)
                weights=batch['valid'].clone()
                if variant.endswith('_tail'):
                    ranks=torch.arange(weights.shape[1])[None]/(weights.sum(1,keepdim=True)-1).clamp(min=1)
                    weights=weights*(1+3*ranks)
                loss=(((p-batch['truth'])**2*weights).sum(1)/weights.sum(1)).mean()
                if not torch.isfinite(loss): raise ValueError('nonfinite loss')
                optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step()
                if epoch%5==0:
                    score=mean_score(evaluate(model,inner_contexts)[0])
                    history.append(dict(epoch=epoch,train_mse=float(loss.detach()),inner_score=score))
                    if score<best:
                        best=score; checkpoint=copy.deepcopy(model.state_dict()); best_epoch=epoch
            model.load_state_dict(checkpoint)
            folder=out/variant/str(seed);folder.mkdir(parents=True)
            torch.save(checkpoint,folder/'model.pt');save(folder/'history.json',history)
            vr,arrays=evaluate(model,validation_contexts)
            tr,_=evaluate(model,train_full)
            # Permutation is an interpretation diagnostic, not used for checkpoint selection.
            shuffled=copy.deepcopy(validation_contexts)
            order=np.random.default_rng(seed).permutation(len(shuffled))
            for i,j in enumerate(order):
                shuffled[i]['statistics']=validation_contexts[j]['statistics'].copy()
                shuffled[i]['conditions']=validation_contexts[j]['conditions'].copy()
            pr,_=evaluate(model,shuffled)
            save(folder/'metrics.json',dict(validation=vr,train=tr,permuted_validation=pr,best_epoch=best_epoch))
            for c in validation_contexts:
                valid=c['valid'].astype(bool)
                arrays[c['cell_id']+'__cycle']=c['query'][valid,0]*processor.query_scale.scaler.scale_[0]*100+processor.query_scale.scaler.mean_[0]*100
            np.savez_compressed(folder/'predictions.npz',**arrays)
            row=dict(variant=variant,seed=seed,best_epoch=best_epoch,seconds=time.perf_counter()-started,
                     parameters=sum(p.numel() for p in model.parameters()),
                     **{k:float(np.mean([r[k] for r in vr])) for k in ('mape','mae_pp','rmse_pp','r2','late_mape','endpoint_error_pp')},
                     train_mape=float(np.mean([r['mape'] for r in tr])),permuted_mape=float(np.mean([r['mape'] for r in pr])))
            summaries.append(row);save(out/'summary.json',summaries)
            print(json.dumps(row),flush=True)
    import csv
    with (out/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    save(out/'completion.json',{'complete':True,'runs':len(summaries),'scope':config['scope']})
    print('OUTPUT='+str(out.resolve()),flush=True)


if __name__=='__main__':
    main()
