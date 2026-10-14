"""Read-only experiment results collection; workbook authoring is separate JS."""
import json,hashlib,subprocess,math
import numpy as np
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/01a0c826-7cec-7363-ae70-be90675e6bf1'
FIELDS=('mape_percent','mae_soh','rmse_soh','r2','late_mape','endpoint_pp','drop_error_pp')
SELECTED=[('MIT','tail_boost_outer/20261005T041231_340759Z','按验证选择的树模型/梯度提升'),('XJTU','tail_boost_outer/20261005T035307_281056Z','按验证选择的树模型/梯度提升'),('Zhu2022 NCA','tail_boost_outer/20261005T040409_096593Z','按验证选择的树模型/梯度提升'),('NASA','nasa_signal_trajectory/ridge/20261004T165029_391663Z','Ridge早期信号轨迹预测')]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def finite(v):return isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v)
def flatten(value,prefix=''):
    if isinstance(value,dict):
        for k,v in value.items():yield from flatten(v,prefix+'.'+k if prefix else k)
    elif isinstance(value,list):
        for i,v in enumerate(value):yield from flatten(v,f'{prefix}[{i}]')
    else:yield prefix,value
def metric_only(value):
    # Keep result statistics, not millions of embedded raw prediction/signal points.
    skip={'prediction','predictions','truth','cycle','capacity','voltage','current','raw','curve','values','label_cycle','early_raw','early_cycle','true','pred','y_true','y_pred','future_cycle','forecast','by_cell','by_cell_seed','per_cell','by_battery'}
    if isinstance(value,dict):return {k:metric_only(v) for k,v in value.items() if k not in skip}
    if isinstance(value,list):return [metric_only(v) for v in value] if any(isinstance(v,(dict,list)) for v in value) else value if len(value)<20 else {'omitted_point_array_length':len(value)}
    return value
def root_for(p):
    for parent in [p.parent,*p.parents]:
        if parent==ROOT/'experiments':break
        if (parent/'config.json').exists():return parent
    return p.parent
def main():
    files=[ROOT/p for p in subprocess.check_output(['rg','--files','experiments','-g','*.json'],cwd=ROOT,text=True,encoding='utf-8').splitlines()]
    metadata={};inventory=[];raw=[];metrics=[];cells=[];runs={};errors=[]
    def meta(p):
        root=root_for(p)
        if root not in metadata:
            cfg=read(root/'config.json') if (root/'config.json').exists() else {}
            dataset=cfg.get('dataset') or (cfg.get('plan',{}).get('dataset') if isinstance(cfg.get('plan',{}),dict) else '') or cfg.get('parent_plan',{}).get('dataset') or ','.join(sorted({str(q.get('dataset','')) for q in cfg.get('plans',[]) if isinstance(q,dict) and q.get('dataset')}))
            verification=[]
            for name in ('independent_verification.json','independent_outer_verification.json','independent_screening_verification.json'):
                if (root/name).exists():
                    d=read(root/name);verification.append(name+':'+str(d.get('passed','未标识')))
            metadata[root]=(root.relative_to(ROOT/'experiments').as_posix(),dataset,'; '.join(verification) or '无同目录独立核验报告')
        return root,metadata[root]
    for p in files:
        name=p.name
        relevant=name in ('summary.json','by_cell.json','by_cell_seed.json','screening_decision.json','selection.json','failure.json','progress.json','metrics.json') or name.endswith('_metrics.json')
        if not relevant or 'source_snapshot' in p.parts:continue
        try:
            d=read(p);root,(run,dataset,proof)=meta(p);source=p.relative_to(root).as_posix();sid=len(inventory)+1;digest=hashlib.sha256(p.read_bytes()).hexdigest();inventory.append([sid,run,dataset,source,digest,proof])
            runs.setdefault(run,[run,dataset,proof,None,None])
            if name=='failure.json':runs[run][3]='失败';runs[run][4]=str(d.get('traceback',d))[-1500:]
            elif name=='progress.json' and runs[run][3]!='失败':runs[run][3]='已完成' if d.get('training_complete',d.get('complete',False)) else '未完成/阶段记录'
            if name in ('summary.json','screening_decision.json','selection.json') or name=='metrics.json' and isinstance(d,dict):
                for key,value in flatten(metric_only(d) if name=='metrics.json' else d):
                    if isinstance(value,(str,int,float,bool)) or value is None:
                        if isinstance(value,str) and len(value)>1200:value=value[:1200]+'（长文本截断）'
                        raw.append([sid,run,dataset,name,key,value])
            elif name in ('by_cell.json','by_cell_seed.json') and isinstance(d,list):
                for i,r in enumerate(d):
                    if not isinstance(r,dict):continue
                    mm=r.get('metrics',r);cells.append([sid,run,dataset,name,r.get('cell_id',r.get('battery_id',str(i))),r.get('seed'),r.get('origin'),*[mm.get(f) if finite(mm.get(f)) else None for f in FIELDS],r.get('joint_pass'),r.get('severe_flat_failure')])
            elif name.endswith('_metrics.json') and isinstance(d,list):
                rr=[r.get('metrics',r) for r in d if isinstance(r,dict)];role=name.removesuffix('_metrics.json');numeric=[]
                for f in FIELDS:
                    vv=[r[f] for r in rr if finite(r.get(f))];numeric.append(sum(vv)/len(vv) if vv else None)
                if any(v is not None for v in numeric):metrics.append([sid,run,dataset,source,role,len(rr),*numeric])
        except Exception as e:errors.append([str(p.relative_to(ROOT)),str(e)])
    current=[];formal=[]
    for dataset,run,model in SELECTED:
        path=ROOT/'experiments'/run;d=read(path/'summary.json');s=d.get('new',d)
        def val(f):
            v=s.get(f);return v.get('mean') if isinstance(v,dict) else v
        def sd(f):
            v=s.get(f);return v.get('seed_sd') if isinstance(v,dict) else None
        mape=val('mape_percent') if 'mape_percent' in s else s.get('mape_mean');n=s.get('cells',d.get('cells'));joint=s.get('joint_pass_fraction');current.append([dataset,model,mape,sd('mape_percent') if 'mape_percent' in s else s.get('mape_sd'),val('rmse_soh'),val('endpoint_pp'),val('drop_error_pp'),n,round(joint*n) if joint is not None and n else None,joint,'未达到全部曲线目标',run])
        if dataset=='NASA':
            from scripts.audit_rate_blend_metrics import metrics as curve_metric,summarize
            records=[]
            for fold in range(1,5):
                for seed in range(20261004,20261009):
                    with np.load(path/f'NASA_fold{fold}'/str(seed)/'predictions.npz') as a:
                        keys={k.split('__100__')[0] for k in a.files if '__100__' in k}
                        for key in keys:records.append(dict(cell_id=key,seed=seed,**curve_metric(*[a[key+'__100__'+f] for f in ('cycle','truth','prediction')])))
            ss,_=summarize(records);row=current[-1];row[4]=ss['rmse_soh']['mean'];row[5]=ss['endpoint_pp']['mean'];row[6]=ss['drop_error_pp']['mean'];row[8]=round(ss['joint_pass_fraction']*n);row[9]=ss['joint_pass_fraction']
    for p in files:
        if p.name!='summary.json':continue
        try:
            d=read(p)
            if not isinstance(d,dict) or not isinstance(d.get('new'),dict) or 'mape_percent' not in d['new']:continue
            root,(run,dataset,proof)=meta(p)
            for label in ('new','baseline','current_primary','fixed_hgb_control'):
                s=d.get(label)
                if not isinstance(s,dict):continue
                def get(f,k='mean'):
                    v=s.get(f);return v.get(k) if isinstance(v,dict) else v
                n=s.get('cells');joint=s.get('joint_pass_fraction');formal.append([run,dataset,label,get('mape_percent'),get('mape_percent','seed_sd'),get('mae_soh'),get('rmse_soh'),get('late_mape'),get('endpoint_pp'),get('drop_error_pp'),n,round(n*joint) if n and joint is not None else None,joint,s.get('endpoint_p90'),s.get('drop_error_p90'),d.get('numeric_gates_passed') if label=='new' else None,proof])
        except Exception as e:errors.append([str(p.relative_to(ROOT)),str(e)])
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),current=current,formal=formal,metric_files=metrics,cell_details=cells,raw_summary=raw,inventory=inventory,runs=list(runs.values()),errors=errors,counts=dict(source_files=len(inventory),formal_rows=len(formal),metric_files=len(metrics),cell_rows=len(cells),raw_fields=len(raw)))
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'collected_results.json').write_text(json.dumps(report,ensure_ascii=False,allow_nan=False),encoding='utf-8');print(json.dumps(report['counts']));print('PARSE_ERRORS='+str(len(errors)))
if __name__=='__main__':main()
