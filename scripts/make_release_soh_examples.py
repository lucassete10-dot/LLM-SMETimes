"""Export documented public replay examples; never call them independent tests."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import shutil
import numpy as np
from battery_smetimes.external_soh import load_context
from scripts.prepare_same_type_forecast import save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);args=p.parse_args();context=load_context(args.package)
    out=Path('results/release_soh_examples')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True);shutil.copy2(__file__,out/'source.py')
    for key in ('b1c11','b1_2C_battery-5','B0005'):
        cell=context['cells'][key];dest=out/key;dest.mkdir()
        np.savez_compressed(dest/'signals.npz',cycle=cell['cycle'],raw_features=cell['raw'])
        take=np.isfinite(cell['soh'])&(cell['soh']>0)&(cell['label_cycle']>100)
        np.savez_compressed(dest/'labels.npz',cycle=cell['label_cycle'][take],soh=cell['soh'][take])
        save_json(dest/'label_info.json',dict(cell_id=key,normalization='initial_comparable_capacity',reference_rule='median_first5_comparable' if cell['metadata']['dataset']=='XJTU' else 'first_comparable',data_role='public development replay, NOT independent new cell'))
    save_json(out/'b1c11/request.json',dict(cell_id='b1c11',specifications=context['registry']['families'][context['cells']['b1c11']['physical_family']]['specifications'],
        operating_conditions=dict(charge_rate_steps_c=[5.4,3.],charge_switch_soc=.5,diagnostic_discharge_rate_c=4.,ambient_temperature_c=30.)))
    save_json(out/'manifest.json',dict(package=str(args.package.resolve()),scope='public development examples only; all these cells trained final release weights'))
    print('OUTPUT='+str(out.resolve()))


if __name__=='__main__':main()
