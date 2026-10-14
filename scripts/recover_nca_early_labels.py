"""Recover omitted <=100-cycle labels for selected development cells only."""
from datetime import datetime,timezone
import json,traceback,zipfile,shutil
from pathlib import Path
import pandas as pd
import numpy as np
from scripts.prepare_zhu2022_nca import COLS,completed_groups,capacity_label
from scripts.train_continuous_length_pilot import NCA_AUDIT
from scripts.prepare_same_type_forecast import sha,save_json


def main():
    run=Path('experiments/nca_length_cv/20261008T064527_966803Z')
    cfg=json.loads((run/'config.json').read_text());ids=cfg['development_cells'];outer=cfg['original_plan']['test']
    assert not set(ids)&set(outer)
    audit=json.loads(NCA_AUDIT.read_text());records={r['cell_id']:r for r in audit['cells']}
    raw=Path(audit['raw_directory']);manifest=json.loads((raw/'intake_manifest.json').read_text())
    archive=raw/manifest['file'];assert manifest['complete'] and sha(archive)==manifest['sha256']==audit['raw_sha256']
    root=Path('data/interim/nca_early_label_recovery')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');root.mkdir(parents=True)
    shutil.copy2(Path(__file__),root/'source_snapshot.py')
    save_json(root/'config.json',dict(development_cells=ids,outer_excluded=outer,raw_sha256=sha(archive),original_audit=str(NCA_AUDIT),
        audit_sha256=sha(NCA_AUDIT),source_sha256=sha(Path(__file__)),capacity_label_source_sha256=sha(Path('scripts/prepare_zhu2022_nca.py')),
        rule='Original measured-current consistency capacity label rule; nominal3500mAh; labels only, never model inputs.',outer_loaded=False))
    entries={};excluded=[];print('RUN='+str(root.resolve()),flush=True)
    try:
        with zipfile.ZipFile(archive) as z:
            for cid in ids:
                record=records[cid];cycles=[];labels=[]
                with z.open(record['member']) as stream:
                    reader=pd.read_csv(stream,usecols=COLS,chunksize=100000)
                    for n,g in completed_groups(reader):
                        if n>100:break
                        try:capacity=capacity_label(g)
                        except ValueError as e:
                            excluded.append(dict(cell_id=cid,cycle=int(n),reason=str(e)));continue
                        cycles.append(n);labels.append(capacity/3500)
                    reader.close()
                assert cycles and np.all(np.diff(cycles)>0)
                assert cycles[0]==record['reference_source_cycle']
                np.testing.assert_allclose(labels[0],record['initial_capacity_mAh']/3500,atol=1e-10,rtol=0)
                path=root/(cid+'.npz');np.savez_compressed(path,cycle=np.asarray(cycles),soh_nominal=np.asarray(labels))
                entries[cid]=dict(path=str(path.resolve()),sha256=sha(path),observations=len(cycles),first=float(cycles[0]),last=float(cycles[-1]))
                save_json(root/'index.json',dict(cells=entries,excluded_records=excluded,outer_loaded=False,complete=False))
                print(cid+' recovered '+str(len(cycles))+' labels',flush=True)
        save_json(root/'index.json',dict(cells=entries,excluded_records=excluded,outer_loaded=False,complete=True))
        save_json(root/'progress.json',dict(terminal=True,success=True,completed_cells=len(entries)))
    except BaseException:
        save_json(root/'failure.json',dict(traceback=traceback.format_exc()));save_json(root/'progress.json',dict(terminal=True,success=False,completed_cells=len(entries)));raise


if __name__=='__main__':main()
