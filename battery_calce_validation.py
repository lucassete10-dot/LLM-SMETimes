"""Pre-specified CALCE cell-held-out and NASA-only transfer evaluation (V/I)."""
import argparse
import hashlib
import json
from pathlib import Path
import csv
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits
from scipy.io import loadmat

from battery_raw_mvp import RawBattery
from battery_smetimes_regularized import Features, RegularizedSMETimes, train
from battery_mvp import compute_metrics
from prepare_calce import IDS, NAMES, vi_features


class VIFeatures(Features):
    def fit(self, batteries, semantic, ids):
        x=np.concatenate([batteries[k].features for k in ids])
        s=np.concatenate([semantic[k] for k in ids])
        self.scaler=StandardScaler().fit(x)
        self.scaler.scale_=np.maximum(self.scaler.scale_,.02)
        self.pca=PCA(n_components=8,svd_solver='randomized',random_state=2026).fit(s)
        self.semantic_scaler=StandardScaler().fit(self.pca.transform(s))
        self.ridge=Ridge(alpha=10).fit(self.scaler.transform(x),np.concatenate([batteries[k].soh for k in ids]))
        return self


def load_batteries():
    batteries={}
    for k in IDS:
        path=Path(r'D:\LLM\CALCE\processed')/f'{k}.npz'
        d=np.load(path)
        assert list(d['names'])==NAMES
        batteries[k]=RawBattery(k,d['observation'],d['observation'],d['features'],d['capacity'],d['soh'],str(path))
    for k in ['B0005','B0006','B0007','B0018']:
        path=Path(r'D:\LLM\NASA\raw\extracted')/f'{k}.mat'
        xx,qq=[],[]
        for record in loadmat(path,simplify_cells=True)[k]['cycle']:
            if record['type']!='discharge':continue
            d=record['data']
            xx.append(vi_features(np.asarray(d['Time']),np.asarray(d['Voltage_measured']),np.asarray(d['Current_measured'])))
            qq.append(float(d['Capacity']))
        x=np.asarray(xx); q=np.asarray(qq,dtype='float32'); numbers=np.arange(1,len(q)+1)
        batteries[k]=RawBattery(k,numbers,numbers,x-x[0],q,q/q[0],str(path))
    return batteries


def prompt(row):
    d=dict(zip(NAMES,row))
    return ("Lithium-ion battery discharge prefix, first 1000 seconds. Changes relative to this cell's first observed discharge: "
            f"initial voltage {d['voltage_v_t0000s']:.5f} V, voltage at 1000 seconds {d['voltage_v_t1000s']:.5f} V, "
            f"mean voltage {d['voltage_mean_v']:.5f} V, mean absolute current {d['current_abs_mean_a']:.5f} A, "
            f"voltage drop {d['voltage_drop_v']:.5f} V, prefix energy {d['prefix_energy_wh']:.5f} Wh.")


@torch.no_grad()
def embeddings(batteries,out):
    from transformers import AutoTokenizer,LlamaForCausalLM
    signature=hashlib.sha256(('vi_prompt_v1').encode()+b''.join(b.features.tobytes() for b in batteries.values())).hexdigest()[:16]
    cache=out/f'vi_semantic_{signature}.pt'
    if cache.exists():return {k:v.float().numpy() for k,v in torch.load(cache,weights_only=True).items()}
    device='cuda' if torch.cuda.is_available() else 'cpu'
    path='models/llm/llama-3.2-1b-instruct'
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='right'
    model=LlamaForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16 if device=='cuda' else torch.float32).to(device).eval()
    result={}
    for key,battery in batteries.items():
        texts=[prompt(row) for row in battery.features]; pieces=[]
        for start in range(0,len(texts),32):
            encoded=tokenizer(texts[start:start+32],padding=True,return_tensors='pt').to(device)
            hidden=model.model(**encoded,use_cache=False).last_hidden_state
            last=encoded['attention_mask'].sum(1)-1
            pieces.append(hidden[torch.arange(len(last),device=device),last].float().cpu())
        result[key]=torch.cat(pieces)
        print('Llama V/I features',key,len(texts),flush=True)
    torch.save(result,cache)
    del model
    if device=='cuda':torch.cuda.empty_cache()
    return {k:v.numpy() for k,v in result.items()}


def fit_protocol(batteries,semantic,development,tests,folder,epochs):
    folder.mkdir(parents=True,exist_ok=True)
    selected={}
    for fusion in (False,True):
        name='smetimes_fusion' if fusion else 'numeric_ablation'
        best=[]
        for val in development:
            train_ids=[k for k in development if k!=val]
            prep=VIFeatures().fit(batteries,semantic,train_ids)
            data={k:prep.transform(batteries[k],semantic[k]) for k in development}
            _,epoch,history=train(data,train_ids,val,fusion,2026,epochs)
            best.append(epoch)
            (folder/f'{name}_validate_{val}.json').write_text(json.dumps(history,indent=2))
        selected[name]=int(np.median(best))
    prep=VIFeatures().fit(batteries,semantic,development)
    joblib.dump(vars(prep),folder/'preprocessing.joblib')
    data={k:prep.transform(batteries[k],semantic[k]) for k in development+tests}
    forest=RandomForestRegressor(n_estimators=300,min_samples_leaf=3,max_features=.8,random_state=2026,n_jobs=-1)
    forest.fit(np.concatenate([batteries[k].features for k in development]),np.concatenate([batteries[k].soh for k in development]))
    joblib.dump(forest,folder/'random_forest.joblib')
    predictions={k:{'ridge':data[k][3].numpy(),'random_forest':forest.predict(batteries[k].features)} for k in tests}
    seed_metrics={k:{} for k in tests}
    for fusion in (False,True):
        name='smetimes_fusion' if fusion else 'numeric_ablation'
        runs={k:[] for k in tests}
        for seed in (2026,2027,2028):
            model,_,_=train(data,development,None,fusion,seed,selected[name])
            torch.save({'state_dict':model.state_dict(),'semantic':fusion,'feature_dim':29,
                        'feature_names':NAMES,'train_ids':development,'epochs':selected[name],'seed':seed},folder/f'{name}_{seed}.pt')
            with torch.no_grad():
                for k in tests:
                    prediction=model(*data[k][:4])[0].numpy()
                    runs[k].append(prediction)
                    seed_metrics[k].setdefault(name,[]).append({'seed':seed,**compute_metrics(prediction[15:],batteries[k].soh[15:])})
        for k in tests:predictions[k][name]=np.mean(runs[k],axis=0)
    results={k:{'training_batteries':development,'selected_epochs':selected,'test_cycles':len(batteries[k].soh)-15,
                'metrics':{name:compute_metrics(pred[15:],batteries[k].soh[15:]) for name,pred in predictions[k].items()},
                'seed_metrics':seed_metrics[k]} for k in tests}
    (folder/'metrics.json').write_text(json.dumps(results,indent=2))
    return results,predictions


def predict_vi(folder,delta_features,semantic_embeddings,fusion=True):
    """Load a saved V/I ensemble; inputs contain no capacity/SOH labels."""
    folder=Path(folder)
    prep=VIFeatures()
    prep.__dict__.update(joblib.load(folder/'preprocessing.joblib'))
    x=np.asarray(delta_features,dtype='float32')
    marks=np.asarray(semantic_embeddings,dtype='float32')
    if x.ndim!=2 or x.shape[1]!=29 or len(x)!=len(marks) or not np.isfinite(x).all() or not np.isfinite(marks).all():
        raise ValueError('Expected finite aligned 29-feature rows and Llama embeddings')
    n=len(x)
    battery=RawBattery('inference',np.arange(n),np.arange(n),x,np.zeros(n),np.zeros(n),'')
    data=prep.transform(battery,marks)
    name='smetimes_fusion' if fusion else 'numeric_ablation'
    runs=[]
    for seed in (2026,2027,2028):
        ckpt=torch.load(folder/f'{name}_{seed}.pt',weights_only=True)
        model=RegularizedSMETimes(fusion,feature_dim=29)
        model.load_state_dict(ckpt['state_dict']);model.eval()
        with torch.no_grad():runs.append(model(*data[:4])[0].numpy())
    return np.mean(runs,axis=0)


def save_plots(batteries,predictions,out,title):
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    rows=[]
    for ax,k in zip(axes.flat,IDS):
        b=batteries[k]
        ax.plot(b.discharge_number[15:],b.soh[15:],'k',label='True SOH',linewidth=1.8)
        for name,pred in predictions[k].items():
            ax.plot(b.discharge_number[15:],pred[15:],label=name,alpha=.8,linewidth=1)
        ax.set(title=k,xlabel='Observed discharge segment',ylabel='SOH')
        ax.grid(alpha=.2)
        for idx in range(15,len(b.soh)):
            rows.append({'cell':k,'observation':int(b.discharge_number[idx]),'true_soh':float(b.soh[idx]),
                         **{n:float(v[idx]) for n,v in predictions[k].items()}})
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out.with_suffix('.png'),dpi=170)
    plt.close(fig)
    with out.with_suffix('.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('battery_calce_results'))
    parser.add_argument('--epochs',type=int,default=120)
    args=parser.parse_args();args.output.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    batteries=load_batteries()
    configuration={'feature_names':NAMES,'prefix_seconds':1000,'temperature':'unavailable; excluded in BOTH datasets',
                   'target':'capacity/first accepted cycle capacity','seeds':[2026,2027,2028],
                   'max_epochs':args.epochs,'architecture':'same regularized SMETimes derivative; input dimension 29',
                   'protocol':'nested leave-one-CALCE-cell-out; NASA-only refit on common V/I features for transfer',
                   'test_selection':'no capacity smoothing/outlier trimming; fixed completeness checks; see extraction audits',
                   'source':'https://calce.umd.edu/battery-data',
                   'source_hashes':{k:hashlib.sha256(Path(b.source_path).read_bytes()).hexdigest() for k,b in batteries.items()}}
    (args.output/'configuration.json').write_text(json.dumps(configuration,indent=2))
    semantic=embeddings(batteries,args.output)
    within,preds={},{}
    with threadpool_limits(limits=4):
        # Transfer is fit/selected using NASA cells only, before CALCE CV metrics.
        nasa=[k for k in batteries if k not in IDS]
        transfer,t_preds=fit_protocol(batteries,semantic,nasa,IDS,args.output/'nasa_only_transfer',args.epochs)
        print('NASA ONLY TRANSFER',json.dumps({k:v['metrics'] for k,v in transfer.items()}),flush=True)
        save_plots(batteries,t_preds,args.output/'nasa_only_transfer','NASA-only training: CALCE transfer, no target-cell fitting')
        for test in IDS:
            result,pred=fit_protocol(batteries,semantic,[k for k in IDS if k!=test],[test],args.output/test,args.epochs)
            within.update(result);preds.update(pred)
            (args.output/'within_calce.json').write_text(json.dumps(within,indent=2))
            print('HELD OUT',test,json.dumps(result[test]['metrics']),flush=True)
        save_plots(batteries,preds,args.output/'calce_held_out','CALCE: nested cell-held-out validation (first 1000 s, voltage/current)')
    summary={}
    for label,results in [('calce_retrained',within),('nasa_only_transfer',transfer)]:
        summary[label]={name:{m:float(np.mean([results[k]['metrics'][name][m] for k in IDS])) for m in ('mae','rmse','r2')}
                        for name in results[IDS[0]]['metrics']}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print('SUMMARY',json.dumps(summary),flush=True)


if __name__=='__main__':main()
