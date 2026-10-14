"""Download the original Severson Batch 1 MAT from a public Kaggle mirror.

The study's canonical data portal is data.matr.io; this program records that
the byte source is a third-party mirror and validates the ZIP before extraction.
"""
import hashlib
import json
from pathlib import Path
import zipfile
import requests

ROOT=Path(r'D:\LLM\MIT\raw')
NAME='2017-05-12_batchdata_updated_struct_errorcorrect.mat'
URL=('https://www.kaggle.com/api/v1/datasets/download/'
     'rickandjoe/mit-battery-degradation-dataset/'+NAME)


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    target=ROOT/(NAME+'.zip')
    partial=ROOT/(NAME+'.zip.partial')
    if not target.exists():
        pos=partial.stat().st_size if partial.exists() else 0
        with requests.get(URL,headers={'Range':f'bytes={pos}-'},stream=True,timeout=(30,120)) as response:
            response.raise_for_status()
            if pos and response.status_code!=206:
                raise RuntimeError('Resume was not honored; refusing to append')
            total=int(response.headers['Content-Range'].split('/')[-1]) if response.status_code==206 else int(response.headers['Content-Length'])
            print('Download bytes',total,'resume',pos,flush=True)
            next_report=pos+100_000_000
            with partial.open('ab' if pos else 'wb') as handle:
                for chunk in response.iter_content(2**20):
                    if not chunk:continue
                    handle.write(chunk)
                    pos+=len(chunk)
                    if pos>=next_report:
                        print('Downloaded',pos,'/',total,flush=True)
                        next_report+=100_000_000
        if pos!=total:raise RuntimeError(f'Incomplete: {pos}/{total}')
        partial.replace(target)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:raise RuntimeError('ZIP CRC failed')
        files=[x for x in archive.infolist() if not x.is_dir()]
        if len(files)!=1 or files[0].filename!=NAME:raise RuntimeError('Unexpected ZIP contents')
        dest=(ROOT/NAME).resolve()
        if not dest.exists():
            with archive.open(files[0]) as src,dest.open('wb') as out:
                for block in iter(lambda:src.read(2**20),b''):
                    out.write(block)
    manifest={'canonical_project':'https://data.matr.io/1/projects/5c48dd2bc625d700019f3204',
              'study_code':'https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation',
              'download_mirror':URL,'zip_size':target.stat().st_size,
              'zip_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
              'mat_size':dest.stat().st_size,
              'mat_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Verified',manifest,flush=True)


if __name__=='__main__':main()
