"""Read primitive-only public HUST records without enabling pickle globals."""
import argparse
import json
import pickle
import zipfile
import numpy as np


class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        raise pickle.UnpicklingError('Disallowed global: '+module+'.'+name)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--member',default='HUST/HUST_1-1.pkl');args=ap.parse_args()
    with zipfile.ZipFile('data/raw/five_dataset_public/BatteryLife_v12/HUST.zip') as z:
        with z.open(args.member) as stream:cell=PrimitiveUnpickler(stream).load()
    meta={k:v for k,v in cell.items() if k!='cycle_data'}
    print(json.dumps(meta,ensure_ascii=False,default=str))
    cycles=cell['cycle_data'];print('CYCLES',len(cycles),'TYPE',type(cycles).__name__)
    if isinstance(cycles,dict):cycles=list(cycles.values())
    for row in [cycles[0],cycles[1],cycles[min(99,len(cycles)-1)],cycles[-1]]:
        print('KEYS',list(row))
        for k,v in row.items():
            if isinstance(v,list):
                a=np.asarray(v,float);print(k,len(a),'min',np.nanmin(a),'max',np.nanmax(a),'first',a[:4].tolist(),'last',a[-4:].tolist())
            else:print(k,v)


if __name__=='__main__':main()
