"""Independent raw-signal audit of deterministic window-feature sample."""
import argparse,json,zipfile
from pathlib import Path
import numpy as np
from scripts.inspect_hust_public import PrimitiveUnpickler
from scripts.prepare_same_type_forecast import sha,save_json


def independent_window(row,stage,low,high):
    t,v,i=[np.asarray(row[k],float) for k in ('time_in_s','voltage_in_V','current_in_A')]
    segments=[]
    for j in range(len(t)-1):
        stage_ok=(i[j]>.1 and i[j+1]>.1) if stage=='charge' else (i[j]<-.1 and i[j+1]<-.1)
        mv=(v[j]+v[j+1])/2
        if stage_ok and low<=mv<high:segments.append((t[j+1]-t[j],abs((i[j]+i[j+1])/2),mv))
    if not segments:return np.array([0,0,np.nan,np.nan,np.nan,np.nan,0])
    a=np.array(segments);duration=sum(a[:,0]);w=a[:,0]/duration
    mi=sum(w*a[:,1]);mv=sum(w*a[:,2])
    return np.array([duration,sum(a[:,0]*a[:,1])/3600,mi,np.sqrt(sum(w*(a[:,1]-mi)**2)),mv,np.sqrt(sum(w*(a[:,2]-mv)**2)),1])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('index',type=Path);path=parser.parse_args().index
    index=json.loads(path.read_text());records={r['cell_id']:r for r in index['cells']}
    sample=sorted(records)[:3]+['HUST_9-5'];verified=0;exclusions=0
    assert len(records)==61 and not index['outer_loaded']
    with zipfile.ZipFile(index['source']) as archive:
        members={Path(m).stem:m for m in archive.namelist() if m.endswith('.pkl')}
        for cid in sample:
            rec=records[cid];assert sha(Path(rec['path']))==rec['sha256']
            with np.load(rec['path'],allow_pickle=False) as cache:
                cycles=cache['physics_cycle'];values=cache['physics_raw']
            with archive.open(members[cid]) as stream:raw=PrimitiveUnpickler(stream).load()
            wanted={1,100,200};excluded={r['cycle'] for r in rec['exclusions']}
            for row in raw['cycle_data']:
                c=float(row['cycle_number'])
                if c in excluded:
                    t=np.asarray(row['time_in_s'],float)
                    assert not np.isfinite(t).all() or np.any(np.diff(t)<=0)
                    exclusions+=1
                if c not in wanted:continue
                j=np.flatnonzero(cycles==c);assert len(j)==1
                expected=np.concatenate([independent_window(row,stage,lo,hi) for stage,windows in index['windows'].items() for lo,hi in windows])
                np.testing.assert_allclose(values[j[0]],expected,rtol=1e-10,atol=1e-10,equal_nan=True)
                verified+=1
    proof=dict(passed=True,index_sha256=sha(path),sample_cells=sample,verified_cycle_vectors=verified,
        invalid_time_exclusions_checked=exclusions,capacity_fields_not_used=True,
        audit_scope='Independent raw replay of fixed 4-cell, 3-cycle sample; not every cached vector.',goal_complete=False)
    save_json(path.parent/'sample_verification.json',proof);print(json.dumps(proof))


if __name__=='__main__':main()
