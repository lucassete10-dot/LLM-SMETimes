"""Passive feature availability inventory; no label-based selection or fitting."""
import json,shutil,time
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import numpy as np,joblib
from battery_smetimes.forecast import charge_rates
from scripts.train_curve_fidelity import AUDIT,load_subset
from scripts.prepare_same_type_forecast import save_json,sha
from prepare_mit import ENHANCED_NAMES
from prepare_xjtu import FEATURE_NAMES

def names(dataset):
    if dataset=='MIT':return ENHANCED_NAMES
    if dataset=='XJTU':return FEATURE_NAMES
    if dataset=='NASA':return ([f'{channel}_{s}s' for channel in ('voltage','current','temperature') for s in range(0,1001,100)]
        +['voltage_drop','voltage_mean','voltage_std','current_abs_mean','current_std','temperature_mean','temperature_std','temperature_change','temperature_max','early_throughput_ah','early_energy_wh'])
    return [f'{channel}_{s}s' for channel in ('voltage','current') for s in range(0,901,90)]+[]

def nca_names():
    return [f'{channel}_{s}s' for channel in ('voltage','current') for s in list(range(0,901,90))+['mean','std','min','max','change']]

def groups(dataset,feature_names):
    def contains(n,text):return text in n.lower()
    return {kind:[i for i,n in enumerate(feature_names) if contains(n,kind)] for kind in ('voltage','current','temperature')}

def inventory(key,dataset,family,status,stratum,c,x,feature_names):
    c=np.asarray(c);x=np.asarray(x,dtype=float);take=c<=100;c,x=c[take],x[take]
    assert x.shape==(len(c),len(feature_names))
    finite=np.isfinite(x);g=groups(dataset,feature_names)
    stat={}
    for name,cols in g.items():
        z=finite[:,cols]
        stat[name]=dict(columns=cols,finite_fraction=float(z.mean()) if z.size else None,
                        completely_missing_columns=sum(not finite[:,i].any() for i in cols))
    return dict(cell_id=key,dataset=dataset,physical_family=family,physical_status=status,
                condition_stratum=stratum,early_records=len(c),first_source_cycle=float(c[0]) if len(c) else None,
                last_source_cycle=float(c[-1]) if len(c) else None,strictly_increasing=bool(np.isfinite(c).all() and np.all(np.diff(c)>0)),
                inf_values=int(np.isinf(x).sum()),nan_values=int(np.isnan(x).sum()),
                all_missing_columns=[i for i in range(x.shape[1]) if not finite[:,i].any()],groups=stat,
                capacity_labels_used=False,future_signals_used=False)

def main():
    root=Path('experiments/personalized_feature_audit')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True)
    print('RUN='+str(root.resolve()),flush=True);snapshot=root/'source_snapshot';snapshot.mkdir();start=time.perf_counter()
    paths=[Path(__file__),Path('prepare_mit.py'),Path('prepare_xjtu.py'),Path('adapt_nasa_mat_v2.py'),Path('scripts/prepare_zhu2022_nca.py'),Path('battery_smetimes/forecast.py'),Path('battery_smetimes/nasa_cycle_calibration.py'),Path('docs/soh_feature_personalized_goal_20261006.md')]
    for p in paths:shutil.copy2(p,snapshot/p.name)
    regpath=AUDIT/'cell_registry.json';phys=AUDIT/'source_snapshot/battery_physical_registry.json'
    records=json.loads(regpath.read_text())['cells'];plans=json.loads((AUDIT/'splits.json').read_text())['plans']
    rows=[];sources={};schemas={}
    for dataset in ('MIT','XJTU','NASA'):
        ids=[k for k,r in records.items() if r['metadata']['dataset']==dataset];cells=load_subset(ids)
        fn=names(dataset);schemas[dataset]=fn
        for key,c in cells.items():
            # Legacy MIT condition_stratum includes batch provenance. Do not
            # count a batch/direction label as a physically defined condition.
            stratum=c['condition_stratum']
            if dataset=='MIT':
                stratum=json.dumps(dict(charge_rates=charge_rates(c['metadata']),charge_protocol=c['metadata']['charge_protocol'].replace('-newstructure',''),diagnostic_discharge_rate_c=c['metadata']['diagnostic_discharge_rate_c']))
            rows.append(inventory(key,dataset,c['physical_family'],c['physical_status'],stratum,c['cycle'],c['raw'],fn));sources[c['prepared_file']]=c['prepared_sha256']
        print('AUDITED '+dataset+' '+str(len(ids)),flush=True)
    old=Path('experiments/zhu2022_nca_forecast/20261004T161453_449950Z/config.json');ncfg=json.loads(old.read_text());prepared=Path(ncfg['prepared_directory'])/'prepared_cells.joblib'
    assert sha(prepared)==ncfg['prepared_sha256'];nca=joblib.load(prepared);sources[str(prepared)]=sha(prepared);schemas['Zhu2022NCA']=nca_names()
    for key,c in nca.items():
        rows.append(inventory(key,'Zhu2022NCA','Zhu2022_dataset1_NCA_18650_3p5Ah_cohort','manufacturer_and_model_unverified',json.dumps(list(c['conditions'])),c['early_cycle'],c['early_raw'],nca_names()))
    nplans=[dict(p,dataset='Zhu2022NCA') for p in ncfg['plans']];plans+=nplans;lookup={r['cell_id']:r for r in rows};support=[]
    for p in plans:
        for role in ('train','validation','test'):
            assert len(p[role])==len(set(p[role]))
        assert not set(p['train'])&set(p['validation']) and not set(p['train']+p['validation'])&set(p['test'])
        counts=Counter(lookup[k]['condition_stratum'] for k in p['train'])
        support.append(dict(fold=p['name'],dataset=p['dataset'],train_counts=dict(counts),local_expert_minimum=6,
                            eligible_local_strata=[k for k,v in counts.items() if v>=6],
                            fallback_strata=[k for k,v in counts.items() if v<6]))
    summary={}
    for dataset in schemas:
        rr=[r for r in rows if r['dataset']==dataset]
        summary[dataset]=dict(cells=len(rr),physical_families=dict(Counter(r['physical_family'] for r in rr)),
           conditions=dict(Counter(r['condition_stratum'] for r in rr)),early_records_range=[min(r['early_records'] for r in rr),max(r['early_records'] for r in rr)],
           cells_with_missing_temperature=sum(r['groups']['temperature']['finite_fraction'] in (None,0.) for r in rr),
           inf_values=sum(r['inf_values'] for r in rr),all_folds_local_eligible_counts=[len(s['eligible_local_strata']) for s in support if s['dataset']==dataset])
    report=dict(summary=summary,cells=rows,schemas=schemas,train_only_local_support=support,passive_inventory=True,
                fitted_statistics=False,model_or_feature_selection=False,capacity_labels_used=False,future_signals_used=False,
                existing_integrity_rules_only=True,new_exclusions=[],goal_complete=False)
    save_json(root/'coverage.json',report)
    save_json(root/'config.json',dict(source_sha256={str(p):sha(p) for p in paths},prepared_sha256=sources,cell_registry_sha256=sha(regpath),physical_registry_sha256=sha(phys),splits_sha256=sha(AUDIT/'splits.json'),nca_config_sha256=sha(old),origin=100))
    save_json(root/'progress.json',dict(complete=True,seconds=time.perf_counter()-start,rows=len(rows),new_exclusions=0,goal_complete=False))
    text=['# 个性化SOH特征覆盖审计（只读盘点）','', '无模型拟合、无标签/未来信号评分、无新增排除。全队列盘点不用于拟合统计或选参；下一轮只能使用训练侧数据。','', '|队列|电芯数|前100源循环有效记录范围|无可用温度电芯|训练侧可建立≥6芯局部专家的工况数（各折）|','|---|---:|---|---:|---|']
    for d,s in summary.items():text.append(f"|{d}|{s['cells']}|{s['early_records_range']}|{s['cells_with_missing_temperature']}|{s['all_folds_local_eligible_counts']}|")
    text+=['','## 下一步约束','','- XJTU诊断循环稀疏：前100源循环不是100条完整诊断记录，跨循环趋势必须使用真实源循环间距。','- NASA家族化学体系/型号未核实且只有4芯，不按截止电压强行建立小样本局部专家。','- NCA现有输入为固定900秒V/I；没有实测T通道，已知环境温度作为工况，不伪造温升特征。','- 既有MIT/NASA和XJTU原始特征包含固定窗I积分Ah/Wh。这不是完整容量标签泄漏，但新版特征消融应明确排除这些Q衍生列作为主输入，对旧基线单独注明；不能偷偷改变旧模型。','- 当前旧prefix已有起始/末期/差分/短窗趋势，缺陷不等于完全没有趋势特征；应做对照实验而不是重复堆特征。','- 数据集/批次不得代替物理家族证据；NASA/XJTU/NCA外部型号识别限制保留。','- 所有指标与源文件SHA见coverage.json/config.json；不因这份盘点新增排除。']
    (root/'audit_report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
