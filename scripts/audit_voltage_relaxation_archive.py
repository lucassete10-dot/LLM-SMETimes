"""Raw-field/schema preview after official archive checksum verification."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
import pandas as pd
from scripts.prepare_same_type_forecast import save_json,sha

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--raw-dir',type=Path,required=True);args=parser.parse_args()
    manifest=json.loads((args.raw_dir/'intake_manifest.json').read_text())
    if not manifest['complete']:raise ValueError('intake incomplete; no training eligibility claim')
    archive=args.raw_dir/manifest['file']
    assert sha(archive)==manifest['sha256']
    out=Path('experiments/voltage_relaxation_audit')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True)
    shutil.copy2(__file__,out/Path(__file__).name)
    rows=[];previews=[]
    with zipfile.ZipFile(archive) as z:
        files=[i for i in z.infolist() if i.filename.lower().endswith('.csv') and not i.is_dir() and '/__MACOSX/' not in i.filename]
        assert len({Path(i.filename).name for i in files})==len(files)
        for i in files:
            with z.open(i) as h:
                frame=pd.read_csv(h,nrows=3)
            columns=list(frame.columns)
            needed={'time/s','Ecell/V','<I>/mA','cycle number','Q discharge/mA.h'}
            rows.append(dict(archive_member=i.filename,uncompressed_bytes=i.file_size,
                columns=columns,required_fields_present=needed.issubset(columns)))
        for i in files[:2]:
            with z.open(i) as h:
                frame=pd.read_csv(h,nrows=200000)
            cycle=pd.to_numeric(frame['cycle number'],errors='coerce')
            previews.append(dict(archive_member=i.filename,read_rows=len(frame),preview_only=True,
                head=frame.head(4).replace({np.nan:None}).to_dict(orient='records'),
                cycle_min=float(cycle.min()),cycle_max=float(cycle.max()),
                unique_cycles=int(cycle.nunique()),time_min=float(frame['time/s'].min()),
                time_max=float(frame['time/s'].max()),current_min_mA=float(frame['<I>/mA'].min()),
                current_max_mA=float(frame['<I>/mA'].max())))
    save_json(out/'audit.json',dict(archive_sha256=manifest['sha256'],license=manifest['license'],
        csv_members=len(rows),files=rows,previews=previews,training_eligible=False,
        reason='schema preview only; full cycle/label extraction and input coverage not yet audited',
        physical_family='Zhu2022_dataset1_NCA_18650_3p5Ah_cohort',manufacturer_model='not yet verified; never pool by NCA label alone',
        raw_directory=str(args.raw_dir.resolve()),official_record_id=manifest['record_id']))
    print('OUTPUT='+str(out.resolve()),'CSV members='+str(len(rows)),flush=True)
    print(json.dumps(previews,ensure_ascii=True)[:5000],flush=True)

if __name__=='__main__':main()
