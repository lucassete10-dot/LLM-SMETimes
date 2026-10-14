"""Cross-compile MCU sources for reference M3/M4 cores; no board flashing."""
import json
from pathlib import Path
import shutil
import subprocess
from scripts.prepare_same_type_forecast import save_json


def main():
    run=Path(json.loads(Path('experiments/stm32_soh/latest.json').read_text())['folder'])
    folder=Path('deployment/stm32_soh')/run.name
    compiler=shutil.which('arm-none-eabi-gcc')
    size=shutil.which('arm-none-eabi-size')
    if not compiler or not size: raise RuntimeError('STM32 ARM GCC tools unavailable')
    reports=[]
    for core in ('cortex-m3','cortex-m4'):
        for domain in ('MIT','XJTU'):
            dest=folder/'arm_objects'/core/domain; dest.mkdir(parents=True,exist_ok=False)
            for source in (folder/'soh_features.c',folder/domain/'soh_model.c'):
                obj=dest/(source.stem+'.o')
                subprocess.run([compiler,'-std=c99','-Os',f'-mcpu={core}','-mthumb','-mfloat-abi=soft',
                    '-ffunction-sections','-fdata-sections','-fstack-usage','-Wall','-Wextra','-Werror',
                    '-c',str(source),'-o',str(obj)],check=True,capture_output=True,text=True)
                report=subprocess.run([size,str(obj)],check=True,capture_output=True,text=True).stdout
                stack=(obj.with_suffix('.su')).read_text()
                reports.append({'domain':domain,'core':core,'source':str(source),'object':str(obj),
                                'size_output':report,'stack_usage_output':stack})
    save_json(folder/'arm_cross_compile.json',{'status':'passed','targets':['cortex-m3','cortex-m4'],
        'compiler':compiler,'optimization':'-Os, soft float, no fast-math',
        'scope':'relocatable objects only; no board startup/linker script or math library included; not total firmware size or board timing',
        'reports':reports})
    print('ARM cross-compilation passed: 8 objects, Cortex-M3/M4, MIT/XJTU')


if __name__=='__main__': main()
