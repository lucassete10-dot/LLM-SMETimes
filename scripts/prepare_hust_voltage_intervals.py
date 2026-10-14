"""Prefix-cycle signal cache for development cells only; no label features."""
from datetime import datetime,timezone
import json,zipfile
from pathlib import Path
import numpy as np
from battery_smetimes.voltage_interval_features import cycle_features,WINDOWS
from scripts.inspect_hust_public import PrimitiveUnpickler
from scripts.prepare_hust_public import digest
from scripts.prepare_same_type_forecast import sha,save_json


def main():
    cfg=json.loads(Path('experiments/hust_length_smoke/20261008T015438_626064Z/config.json').read_text())
    dev=set(cfg['train']+cfg['validation']); assert len(dev)==61 and not dev&set(cfg['test'])
    source=Path('data/raw/five_dataset_public/BatteryLife_v12/HUST.zip')
    assert digest(source)=='de6b6d1b0b20616fbb96c72f3231c082'
    root=Path('data/interim/hust_voltage_intervals')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'); root.mkdir(parents=True)
    records=[]; print('RUN='+str(root.resolve()),flush=True)
    with zipfile.ZipFile(source) as archive:
        for member in archive.namelist():
            if not member.endswith('.pkl') or Path(member).stem not in dev: continue
            with archive.open(member) as stream: cell=PrimitiveUnpickler(stream).load()
            assert cell['cell_id'] in dev
            cycles=[]; features=[]; excluded=[]
            for row in cell['cycle_data']:
                c=float(row['cycle_number'])
                if c>200: continue
                try:
                    # Redact all unapproved fields before feature computation.
                    f=cycle_features({k:row[k] for k in ('time_in_s','voltage_in_V','current_in_A')})
                    cycles.append(c); features.append(f)
                except ValueError as e: excluded.append(dict(cycle=c,reason=str(e),role='signal',not_bad_cell=True))
            assert len(cycles)>=3 and np.all(np.diff(cycles)>0)
            path=root/(cell['cell_id']+'.npz')
            np.savez_compressed(path,physics_cycle=np.array(cycles),physics_raw=np.array(features))
            records.append(dict(cell_id=cell['cell_id'],path=str(path.resolve()),sha256=sha(path),observations=len(cycles),exclusions=excluded))
            print('CELL',cell['cell_id'],len(cycles),flush=True)
    assert {r['cell_id'] for r in records}==dev
    save_json(root/'index.json',dict(cells=records,source_md5='de6b6d1b0b20616fbb96c72f3231c082',source=str(source.resolve()),
        windows=WINDOWS,feature_schema='Per charge/discharge voltage window: observed duration, partial current-integral Ah, mean/std current, mean/std voltage, availability.',
        full_capacity_or_label_inputs=False,outer_loaded=False,cache_max_source_cycle=200,
        feature_source_sha256=sha(Path('battery_smetimes/voltage_interval_features.py')),goal_complete=False))


if __name__=='__main__':main()
