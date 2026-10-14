"""Link every single-family C pipeline with libm/libc/softfloat at fixed budgets."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
from scripts.prepare_same_type_forecast import save_json,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--package',type=Path,required=True);args=parser.parse_args()
    out=Path('deployment/stm32_linked_reference')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True)
    shutil.copy2(__file__,out/'build_source.py');compiler=shutil.which('arm-none-eabi-gcc');size=shutil.which('arm-none-eabi-size')
    if not compiler or not size:raise RuntimeError('ARM tools unavailable')
    version=subprocess.run([compiler,'--version'],check=True,capture_output=True,text=True).stdout
    common=Path('deployment/common');reports=[]
    for domain,family in [('MIT',1),('XJTU',3),('NASA',2)]:
        dest=out/domain;dest.mkdir();package=args.package/domain
        for name in ('soh_tree.c','soh_tree.h','soh_stream.c','soh_stream.h'):shutil.copy2(package/name,dest/name)
        optional_model=['soh_model'] if (package/'soh_model.c').exists() else []
        for name in optional_model:
            for suffix in ('.c','.h'):shutil.copy2(package/(name+suffix),dest/(name+suffix))
        for name in ('soh_waveform.c','soh_waveform.h','soh_reference_app.c','soh_reference_app.h','soh_reference.ld'):shutil.copy2(common/name,dest/name)
        contract=json.loads((package/'input_contract.json').read_text());cols=','.join(map(str,contract['raw_columns']))
        # Generated model metadata, not manually edited source.
        (dest/'soh_selected_columns.h').write_text(f'#define SOH_FAMILY {family}\nstatic const int soh_raw_columns[]={{{cols}}};\n',encoding='utf-8')
        flags=['-std=c99','-Os','-mcpu=cortex-m4','-mthumb','-mfloat-abi=soft','-ffunction-sections','-fdata-sections','-fstack-usage','-Wall','-Wextra','-Werror']
        objects=[];commands=[]
        try:
            for name in ('soh_tree','soh_stream','soh_waveform','soh_reference_app',*optional_model):
                target=dest/(name+'.o');command=[compiler,*flags,'-c',str(dest/(name+'.c')),'-o',str(target)]
                commands.append(command);subprocess.run(command,check=True,capture_output=True,text=True);objects.append(target)
            command=[compiler,'-mcpu=cortex-m4','-mthumb','-mfloat-abi=soft','-nostartfiles','--specs=nosys.specs',
                '-Wl,--gc-sections',f'-Wl,-Map={dest/"firmware.map"}',f'-T{dest/"soh_reference.ld"}',*[str(p) for p in objects],'-lm','-lc','-o',str(dest/'firmware.elf')]
            commands.append(command);proc=subprocess.run(command,check=True,capture_output=True,text=True)
            sizes=subprocess.run([size,'-A',str(dest/'firmware.elf')],check=True,capture_output=True,text=True).stdout
            sections={}
            for line in sizes.splitlines():
                fields=line.split()
                if len(fields)>=3 and fields[0].startswith('.'):
                    sections[fields[0]]=dict(bytes=int(fields[1]),address=int(fields[2]))
            flash=sum(r['bytes'] for r in sections.values() if 0x08000000<=r['address']<0x08010000)+sections.get('.data',{}).get('bytes',0)
            static_ram=sum(r['bytes'] for r in sections.values() if 0x20000000<=r['address']<0x20004000)
            report=dict(dataset=domain,status='linked',flash_bytes=flash,ram_reserved_bytes=static_ram,sections=sections,compiler_output=proc.stdout+proc.stderr,
                        stack_usage={p.name:p.read_text() for p in dest.glob('*.su')},within_reference_budgets=flash<=65536 and static_ram<=16384)
        except subprocess.CalledProcessError as exc:
            report=dict(dataset=domain,status='failed',command=exc.cmd,stdout=exc.stdout,stderr=exc.stderr)
        report['commands']=commands;save_json(dest/'build_report.json',report);reports.append(report)
        print(domain,report['status'],report.get('flash_bytes'),report.get('ram_reserved_bytes'),flush=True)
    save_json(out/'manifest.json',dict(compiler=compiler,version=version,reports=reports,package=str(args.package.resolve()),
       files_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},
       scope='Reference Cortex-M4 softfloat, 64KiB Flash/16KiB RAM, static buffers and 2KiB reserved stack; no heap, ADC/HAL/clock/board interrupts, real runtime or measured peak stack.',
       pending=['specific STM32 board memory map and startup','sensor/charger event integration','measured peak stack and interrupt headroom','board query latency','new-cell independent evaluation']))
    print('OUTPUT='+str(out.resolve()),flush=True)


if __name__=='__main__':main()
