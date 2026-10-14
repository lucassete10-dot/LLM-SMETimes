"""Discover official Oxford files and save a new, checksum-recorded intake."""
from datetime import datetime,timezone
from html.parser import HTMLParser
import argparse
import hashlib
from pathlib import Path
from urllib.parse import urljoin
import requests
from scripts.prepare_same_type_forecast import save_json

URL='https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac'


class Links(HTMLParser):
    def __init__(self):super().__init__();self.href=None;self.text='';self.links=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a':self.href=dict(attrs).get('href');self.text=''
    def handle_data(self,data):
        if self.href:self.text+=data
    def handle_endtag(self,tag):
        if tag=='a' and self.href:self.links.append((self.text.strip(),urljoin(URL,self.href)));self.href=None


def main():
    p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');args=p.parse_args()
    out=Path('data/raw/oxford_degradation')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');out.mkdir(parents=True)
    r=requests.get(URL,timeout=30);r.raise_for_status();html=Links();html.feed(r.text)
    files=[(name,url) for name,url in html.links if any(s.lower() in name.lower() for s in ('Oxford_Battery_Degradation_Dataset_1.mat','ExampleDC_C1.mat','Readme.txt'))]
    save_json(out/'discovery.json',dict(record_url=URL,links=files,training_eligible=False,
        caveat='Official archive, 8 cells; characterization every 100 source cycles. Cannot pretend 100 diagnostic observations occur before source cycle100. Original license/readme must be retained and checked.'))
    print(files,flush=True)
    if args.download:
        reports=[]
        for name,url in files:
            filename=next(s for s in ('Oxford_Battery_Degradation_Dataset_1.mat','ExampleDC_C1.mat','Readme.txt') if s.lower() in name.lower())
            dest=out/filename
            if dest.exists():continue
            response=requests.get(url,stream=True,timeout=(30,90));response.raise_for_status();digest=hashlib.sha256();size=0
            with dest.open('xb') as handle:
                for chunk in response.iter_content(1024*1024):
                    handle.write(chunk);digest.update(chunk);size+=len(chunk)
            if filename.endswith('.mat'):
                with dest.open('rb') as handle:head=handle.read(128)
                if not (head.startswith(b'MATLAB') or head.startswith(b'\x89HDF')):raise ValueError('download is not MATLAB/HDF data')
            reports.append(dict(name=filename,url=url,bytes=size,sha256=digest.hexdigest()))
            save_json(out/'intake_manifest.json',dict(complete=False,files=reports,training_eligible=False))
            print('SAVED',filename,size,flush=True)
        save_json(out/'intake_manifest.json',dict(complete=bool(reports),files=reports,training_eligible=False))
    print('OUTPUT='+str(out.resolve()),flush=True)


if __name__=='__main__':main()
