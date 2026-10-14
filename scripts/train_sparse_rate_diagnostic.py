"""Same fixed source-window diagnostic with explicit sparse-observation policy."""
from pathlib import Path
import shutil
import numpy as np
from scripts import train_early_rate_diagnostic as runner
from scripts.train_continuous_length_pilot import target
from scripts.prepare_same_type_forecast import sha


def sparse_label(cell):
    c,y=target(cell,100);mask=(c>=101)&(c<=125);c,y=c[mask],y[mask]
    if len(c)<3 or c[-1]-c[0]<15:
        raise ValueError('less than three observed labels spanning15 source cycles in101..125')
    x=c-c.mean()
    return float(-100*x@(y-y.mean())/(x@x)),dict(points=len(c),first=float(c[0]),last=float(c[-1]),sparse=len(c)<10)


def main():
    original_save=runner.save_json
    driver=Path(__file__)
    def record(path,data):
        if Path(path).name=='config.json':
            data=dict(data)
            data['source_sha256'][str(driver)]=sha(driver)
            data['label_policy']=dict(minimum_points=3,minimum_span=15,window=[101,125],interpolated=False)
            data['limitation']='Near-term rate diagnostic; sparse OLS has higher uncertainty; not full-curve or knee evidence.'
            shutil.copy2(driver,Path(path).parent/'source_snapshot'/driver.name)
        original_save(path,data)
    runner.label=sparse_label
    runner.save_json=record
    runner.main()


if __name__=='__main__':main()
