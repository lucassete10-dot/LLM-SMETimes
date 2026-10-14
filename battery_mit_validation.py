"""MIT Batch 1: prior-data prediction, then cell-held-out model adaptation.

All feature and model choices are fixed before MIT test-cell fitting. The MIT
test cells never enter preprocessing, training or epoch selection.
"""
import copy
import csv
import json
from pathlib import Path
import argparse

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from scipy.ndimage import median_filter
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler

from battery_mvp import compute_metrics, set_seed
from battery_raw_mvp import RawBattery
from battery_smetimes_regularized import RegularizedSMETimes, train
from prepare_mit import OUT as MIT_OUT, COMMON_GRID, COMMON_NAMES, ENHANCED_NAMES, feature_vector

RESULT=Path('battery_mit_results')


def battery(name,features,capacity,cycle,source):
    x=np.asarray(features,dtype='float32');q=np.asarray(capacity,dtype='float32')
    return RawBattery(name,np.asarray(cycle),np.asarray(cycle),x,q,q/q[0],str(source))


def load_mit():
    common,rich={},{ }
    for path in sorted(MIT_OUT.glob('b1c*.npz'),key=lambda p:int(p.stem[3:])):
        with np.load(path) as d:
            if list(d['common_names'])!=COMMON_NAMES or list(d['enhanced_names'])!=ENHANCED_NAMES:
                raise ValueError('Feature schema changed: '+str(path))
            common[path.stem]=battery(path.stem,d['common'],d['capacity'],d['cycle'],path)
            rich[path.stem]=battery(path.stem,d['enhanced'],d['capacity'],d['cycle'],path)
    if len(common)!=46:raise ValueError(f'Expected 46 MIT cells; got {len(common)}')
    return common,rich


def calce_to_300(path):
    with np.load(path) as d:
        raw=d['raw_features'];q=d['capacity'];soh=d['soh'];cycle=d['observation']
        rows=[]
        for row in raw:
            v=row[:11:1][:4];i=row[11:22][:4]
            rows.append(feature_vector(COMMON_GRID/60,i,v,np.full(4,np.nan),COMMON_GRID))
        x=np.stack(rows)
    # Source is already audited; preserve its SOH definition and original cell reference.
    return RawBattery(path.stem,cycle,cycle,x-x[0],q,soh,str(path))


def nasa_to_300(path):
    b=loadmat(path,simplify_cells=True)[path.stem]
    rows=[];q=[]
    for rec in b['cycle']:
        if rec['type']!='discharge':continue
        d=rec['data'];t=np.asarray(d['Time']).reshape(-1)
        mask=t<=300
        if mask.sum()<3:raise ValueError('NASA prefix too short')
        rows.append(feature_vector(t[mask]/60,
          np.asarray(d['Current_measured']).reshape(-1)[mask],
          np.asarray(d['Voltage_measured']).reshape(-1)[mask],
          np.full(mask.sum(),np.nan),COMMON_GRID))
        q.append(float(d['Capacity']))
    x=np.stack(rows);q=np.array(q,dtype='float32');cycle=np.arange(1,len(q)+1)
    return battery(path.stem,x-x[0],q,cycle,path)


def load_prior():
    batteries={}
    for path in sorted(Path(r'D:\LLM\CALCE\processed').glob('CS2_*.npz')):
        batteries[path.stem]=calce_to_300(path)
    for path in sorted(Path(r'D:\LLM\NASA\raw\extracted').glob('B*.mat')):
        batteries[path.stem]=nasa_to_300(path)
    assert len(batteries)==8
    return batteries


class NumericPrep:
    def __init__(self,robust=False):self.robust=robust
    def fit(self,batteries,ids):
        x=np.concatenate([batteries[k].features for k in ids])
        self.imputer=SimpleImputer(strategy='median',keep_empty_features=True).fit(x)
        transformed=self.imputer.transform(x)
        self.scaler=(RobustScaler(quantile_range=(10,90)) if self.robust else StandardScaler()).fit(transformed)
        self.scaler.scale_=np.maximum(self.scaler.scale_,.02)
        weights=np.concatenate([np.full(len(batteries[k].soh),1/len(batteries[k].soh)) for k in ids])
        self.ridge=Ridge(alpha=10).fit(self.scaler.transform(transformed),
                   np.concatenate([batteries[k].soh for k in ids]),sample_weight=weights*len(x)/len(ids))
        return self
    def transform(self,battery):
        x=self.scaler.transform(self.imputer.transform(battery.features)).astype('float32')
        x=np.clip(x,-6,6)
        history=np.stack([x[max(0,j-7):j+1].mean(0) for j in range(len(x))])
        base=self.ridge.predict(x).astype('float32')
        return [torch.tensor(v,dtype=torch.float32) for v in
                (x,history,np.zeros((len(x),8),dtype='float32'),base,battery.soh)]


def make_data(prep,batteries,ids):return {k:prep.transform(batteries[k]) for k in ids}


def metrics_by_cell(batteries,preds,ids):
    by_cell={}
    for k in ids:
        truth=batteries[k].soh[15:]
        by_cell[k]={name:compute_metrics(preds[k][name][15:],truth) for name in preds[k]}
    names=list(preds[ids[0]])
    macro={name:{metric:float(np.mean([by_cell[k][name][metric] for k in ids]))
          for metric in ('mae','rmse','r2')} for name in names}
    pooled={name:compute_metrics(np.concatenate([preds[k][name][15:] for k in ids]),
                                  np.concatenate([batteries[k].soh[15:] for k in ids])) for name in names}
    return {'macro':macro,'pooled':pooled,'by_cell':by_cell,
            'cells':len(ids),'observations':sum(len(batteries[k].soh)-15 for k in ids)}


def robust_fit(data,batteries,ids,val_id,seed,epochs=100,patience=12,robust=True):
    set_seed(seed)
    model=RegularizedSMETimes(False,feature_dim=data[ids[0]][0].shape[1])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.03)
    batches=[];weights=[];counts={'downweighted':0,'training_observations':0}
    for key in ids:
        n=len(data[key][0]);sub=np.arange(0,n,3)
        sample=[item[sub] for item in data[key]]
        batches.append(sample)
        median=median_filter(batteries[key].soh,size=9,mode='nearest')
        outlier=np.abs(batteries[key].soh-median)>.04
        counts['downweighted']+=int(outlier[sub].sum())
        counts['training_observations']+=len(sub)
        w=(np.where(outlier[sub],.25,1.) if robust else np.ones(len(sub))).astype('float32')
        w=w/w.sum()/len(ids)
        weights.append(torch.tensor(w))
    x,h,s,base,y=[torch.cat([part[j] for part in batches]) for j in range(5)]
    weight=torch.cat(weights)
    best=None;best_score=float('inf');best_epoch=0;stale=0;history=[]
    for epoch in range(1,epochs+1):
        model.train()
        prediction,reconstructed,_=model(x+.015*torch.randn_like(x),h,s,base)
        if robust:
            error=prediction-y
            absolute=error.abs()
            per=torch.where(absolute<=.025,.5*error.square(),.025*(absolute-.0125))
        else:per=(prediction-y).square()
        loss=(per*weight).sum()+.0002*(reconstructed-x).square().mean()+.01*(prediction-base).square().mean()
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        model.eval()
        with torch.no_grad():
            val=model(*data[val_id][:4])[0]
            score=(val[15:]-data[val_id][4][15:]).abs().mean().item()
        history.append({'epoch':epoch,'train_loss':float(loss.item()),'val_mae':score})
        if score<best_score-1e-6:
            best,best_score,best_epoch,stale=copy.deepcopy(model.state_dict()),score,epoch,0
        else:stale+=1
        if stale>=patience:break
    model.load_state_dict(best);model.eval()
    return model,best_epoch,history,counts


def predict_model(model,data,ids):
    with torch.no_grad():return {k:model(*data[k][:4])[0].numpy() for k in ids}


def save_predictions(path,batteries,preds,ids):
    rows=[]
    for k in ids:
        b=batteries[k]
        for j in range(15,len(b.soh)):
            rows.append({'cell':k,'cycle':int(b.discharge_number[j]),'true_soh':float(b.soh[j]),
                         **{n:float(preds[k][n][j]) for n in preds[k]}})
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def plot_examples(path,batteries,preds,ids,title):
    chosen=ids[:4]
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for ax,k in zip(axes.flat,chosen):
        b=batteries[k];ax.plot(b.discharge_number[15:],b.soh[15:],'k',lw=1.8,label='True SOH')
        for name,values in preds[k].items():
            ax.plot(b.discharge_number[15:],values[15:],lw=1,alpha=.8,label=name)
        ax.set(title=k,xlabel='Observed cycle',ylabel='SOH');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(title);fig.tight_layout();fig.savefig(path,dpi=170);plt.close(fig)


def prior_prediction(prior,mit,out):
    folder=out/'before_mit_training';folder.mkdir(exist_ok=True)
    train_ids=[k for k in prior if k!='B0018']
    val_id='B0018'
    prep=NumericPrep().fit(prior,train_ids)
    joblib.dump(vars(prep),folder/'preprocessing.joblib')
    data=make_data(prep,{**prior,**mit},train_ids+[val_id]+list(mit))
    model,_,history=train(data,train_ids,val_id,False,2026,100,12)
    torch.save({'state_dict':model.state_dict(),'feature_dim':15,'train_ids':train_ids,
                'val_id':val_id,'history':history},folder/'numeric_prior.pt')
    rf=RandomForestRegressor(n_estimators=300,min_samples_leaf=3,max_features=.8,random_state=2026,n_jobs=-1)
    rf.fit(np.concatenate([prior[k].features for k in train_ids]),
           np.concatenate([prior[k].soh for k in train_ids]))
    joblib.dump(rf,folder/'random_forest_prior.joblib')
    predictions={k:{'ridge':data[k][3].numpy(),
                    'random_forest':rf.predict(mit[k].features),
                    'smetimes_numeric':p} for k,p in predict_model(model,data,list(mit)).items()}
    summary=metrics_by_cell(mit,predictions,list(mit))
    (folder/'metrics.json').write_text(json.dumps(summary,indent=2))
    save_predictions(folder/'predictions.csv',mit,predictions,list(mit))
    plot_examples(folder/'examples.png',mit,predictions,list(mit)[:4],
                  'MIT: NASA + CALCE training only, first 300 s')
    print('BEFORE MIT TRAINING',json.dumps(summary['macro']),flush=True)
    return summary


def mit_retraining(common,enhanced,out):
    folder=out/'mit_cell_holdout';folder.mkdir(exist_ok=True)
    ids=list(common)
    dev,test=train_test_split(ids,test_size=.2,random_state=2026,shuffle=True)
    train_ids,val=train_test_split(dev,test_size=.2,random_state=2026,shuffle=True)
    train_ids,val,test=[sorted(group,key=lambda x:int(x[3:])) for group in (train_ids,val,test)]
    val_id=val[0]
    # Remaining validation cells help test epoch stability without entering training.
    split={'train':train_ids,'validation':val,'test':test,'random_state':2026}
    (folder/'split.json').write_text(json.dumps(split,indent=2))
    baseprep=NumericPrep().fit(common,train_ids)
    richprep=NumericPrep(robust=True).fit(enhanced,train_ids)
    joblib.dump(vars(baseprep),folder/'common_preprocessing.joblib')
    joblib.dump(vars(richprep),folder/'enhanced_preprocessing.joblib')
    base=make_data(baseprep,common,ids)
    rich=make_data(richprep,enhanced,ids)
    x_base=np.concatenate([common[k].features for k in train_ids])
    x_rich=richprep.imputer.transform(np.concatenate([enhanced[k].features for k in train_ids]))
    target=np.concatenate([common[k].soh for k in train_ids])
    forest=RandomForestRegressor(n_estimators=300,min_samples_leaf=3,max_features=.8,
                                 random_state=2026,n_jobs=-1).fit(x_base,target)
    trees=ExtraTreesRegressor(n_estimators=200,min_samples_leaf=5,max_features=.8,
                              random_state=2026,n_jobs=-1).fit(x_rich,target)
    joblib.dump(forest,folder/'common_random_forest.joblib')
    joblib.dump(trees,folder/'enhanced_extra_trees.joblib')
    preds={k:{'common_ridge':base[k][3].numpy(),
              'common_rf':forest.predict(common[k].features),
              'enhanced_extra_trees':trees.predict(richprep.imputer.transform(enhanced[k].features))}
           for k in test}
    for name,data,batteries,robust in [('common_numeric',base,common,False),
                                        ('enhanced_mse',rich,enhanced,False),
                                        ('enhanced_robust',rich,enhanced,True)]:
        runs=[];training=[]
        seeds=[2026,2027,2028] if name!='enhanced_mse' else [2026]
        for seed in seeds:
            if name=='common_numeric':
                model,epoch,history=train(data,train_ids,val_id,False,seed,100,12)
                info={'best_epoch':epoch}
            else:
                model,epoch,history,counts=robust_fit(data,batteries,train_ids,val_id,seed,100,12,robust)
                info={'best_epoch':epoch,'outlier_training_counts':counts}
            run=predict_model(model,data,test)
            runs.append(run)
            torch.save({'state_dict':model.state_dict(),'feature_dim':data[test[0]][0].shape[1],
                        'train_ids':train_ids,'val_id':val_id,'seed':seed,'info':info},
                        folder/f'{name}_{seed}.pt')
            training.append({'seed':seed,**info,'history':history})
        for k in test:preds[k][name]=np.mean([run[k] for run in runs],axis=0)
        (folder/f'{name}_training.json').write_text(json.dumps(training,indent=2))
        print('MIT MODEL',name,'done',flush=True)
    metrics=metrics_by_cell(common,preds,test)
    (folder/'metrics.json').write_text(json.dumps(metrics,indent=2))
    save_predictions(folder/'predictions.csv',common,preds,test)
    plot_examples(folder/'examples.png',common,preds,test,
                  'MIT unseen cells: retrained on MIT cells, first 300 s')
    print('AFTER MIT TRAINING',json.dumps(metrics['macro']),flush=True)
    return metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=RESULT);args=p.parse_args()
    args.output.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    common,enhanced=load_mit()
    prior=load_prior()
    config={'mit_cells':len(common),'mit_source':'Severson Batch 1 original MAT from Kaggle mirror',
            'prefix_seconds':300,'common_features':COMMON_NAMES,'enhanced_features':ENHANCED_NAMES,
            'protocol':'NASA+CALCE-only transfer before MIT training; then random 2026 cell split MIT train/val/test',
            'target':'current-cycle discharge capacity / first valid-cycle capacity',
            'target_only':'QDischarge and full discharge duration excluded from features',
            'abnormal_training_rule':'training cell SOH deviation >0.04 from 9-cycle local median gets 0.25 weight only in robust loss; tests unchanged',
            'validation_rule':'epoch chosen on one validation cell; 7 more validation cells remain unused for model choice',
            'no_llama':'numeric SMETimes derivative used; common direct transfer cannot use 1000s frozen Llama checkpoint at 4C',
            'model_seeds':[2026,2027,2028]}
    (args.output/'configuration.json').write_text(json.dumps(config,indent=2))
    before=prior_prediction(prior,common,args.output)
    after=mit_retraining(common,enhanced,args.output)
    (args.output/'summary.json').write_text(json.dumps({'before_mit_training':before['macro'],
                  'after_mit_training':after['macro'],'test_cells':after['cells'],
                  'test_observations':after['observations']},indent=2))


if __name__=='__main__':main()
