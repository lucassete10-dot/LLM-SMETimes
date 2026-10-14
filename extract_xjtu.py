"""Extract only the 55 original MAT files and English protocol PDF."""
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

from download_xjtu import ROOT


def main():
    source = ROOT / 'Battery Dataset.zip'
    output = ROOT / 'extracted'
    output.mkdir(parents=True, exist_ok=True)
    selected = []
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError(f'Unsafe archive entry: {member.filename}')
            if relative.suffix.lower() != '.mat' and relative.name != 'Data Introduction-English.pdf':
                continue
            target = output.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(output.resolve()):
                raise ValueError(f'Archive target escapes output: {target}')
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == member.file_size:
                print('Existing', target.name, flush=True)
            else:
                temporary = target.with_suffix(target.suffix + '.partial')
                with archive.open(member) as input_file, temporary.open('wb') as output_file:
                    shutil.copyfileobj(input_file, output_file, length=2**20)
                if temporary.stat().st_size != member.file_size:
                    raise RuntimeError(f'Extracted size mismatch: {target}')
                temporary.replace(target)
                print('Extracted', target.name, flush=True)
            selected.append({'name': member.filename, 'bytes': member.file_size, 'path': str(target)})
    battery_files = [x for x in selected if x['name'].lower().endswith('.mat')]
    if len(battery_files) != 56:
        raise ValueError(f'Expected 55 battery MAT + 1 compensation MAT; got {len(battery_files)}')
    (output / 'manifest.json').write_text(json.dumps({'source': str(source), 'files': selected}, indent=2))
    print('Extracted MAT files', len(battery_files), flush=True)


if __name__ == '__main__':
    main()
