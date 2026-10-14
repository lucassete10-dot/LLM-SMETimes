"""Inner validation reference support diagnostic, never outer-based selection."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from battery_smetimes.conditional_drift import prefix
from battery_smetimes.tail_boost import conditions
from scripts.train_tail_boost import load_cells
from scripts.train_crossfit_drift_smoke import clean
from scripts.prepare_same_type_forecast import sha,save_json

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);root=parser.parse_args().run;cfg=json.loads((root/'config.json').read_text());assert json.loads((root/'independent_verification.json').read_text())['passed'];rows=[];evidence=[]
    for plan in cfg['plans']:
        cells=load_cells(plan);assert not set(cells)&set(plan['test']);folder=root/plan['name'];b=joblib.load(folder/'components.joblib')['analog'];ids=plan['train'];assert b['train_ids']==ids
        selection=json.loads((folder/'selection.json').read_text());selected=cfg['candidates'][selection['selected']['candidate']]
        file=folder/'validation_components.npz';before=sha(file)
        with np.load(file) as arrays:
            for key in plan['validation']:
                query=clean(cells[key]);c,y,p,q=[arrays[key+'__100__'+f] for f in ('cycle','truth','primary','neighbor'+str(selected['k']))]
                z=b['scaler'].transform(b['imputer'].transform(prefix(query)[None]))[0];local=[i for i,r in enumerate(b['references']) if r['condition']==conditions(query)];pool=local if len(local)>=3 else list(range(len(ids)));order=sorted(pool,key=lambda i:(float(np.linalg.norm(z-b['z'][i])),ids[i]))[:selected['k']]
                references=[b['references'][i] for i in order];outside=np.stack([(c<r['cycle'][0])|(c>r['cycle'][-1]) for r in references]);late=c>=(c[0]+c[-1])/2
                rows.append(dict(fold=plan['name'],cell_id=key,active_weight=selected['weight'],k=selected['k'],reference_ids=[r['cell_id'] for r in references],reference_last_cycles=[float(r['cycle'][-1]) for r in references],point_peer_extrapolation_fraction=float(outside.mean()),late_peer_extrapolation_fraction=float(outside[:,late].mean()),endpoint_extrapolated_peer_fraction=float(outside[:,-1].mean()),reference_endpoint_signed_pp=float(100*(q[-1]-y[-1])),primary_endpoint_signed_pp=float(100*(p[-1]-y[-1])),reference_endpoint_abs_pp=float(100*abs(q[-1]-y[-1])),primary_endpoint_abs_pp=float(100*abs(p[-1]-y[-1])),reference_worse_endpoint=bool(abs(q[-1]-y[-1])>abs(p[-1]-y[-1]))))
        assert sha(file)==before;evidence.append(dict(file=str(file.resolve()),sha256=before,model_sha256=sha(folder/'components.joblib'),selection_sha256=sha(folder/'selection.json')))
    summary={}
    for name,rr in [('all',rows),('active',[r for r in rows if r['active_weight']>0]),('active_extrapolated',[r for r in rows if r['active_weight']>0 and r['endpoint_extrapolated_peer_fraction']>0]),('active_supported',[r for r in rows if r['active_weight']>0 and r['endpoint_extrapolated_peer_fraction']==0])]:
        summary[name]=dict(cells=len(rr),**{f:float(np.mean([r[f] for r in rr])) if rr else None for f in ('point_peer_extrapolation_fraction','late_peer_extrapolation_fraction','endpoint_extrapolated_peer_fraction','reference_endpoint_abs_pp','primary_endpoint_abs_pp','reference_worse_endpoint')})
    save_json(root/'validation_reference_support.json',dict(summary=summary,rows=rows,evidence=evidence,source_sha256=sha(__file__),outer_loaded=False,used_for_model_selection=False,goal_complete=False,interpretation='Validation-only support of selected training references. Endpoint range used only for scoring diagnostics, not to select references. Group means are descriptive, different cells and potentially small counts; not proof extrapolation causes errors.'))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
