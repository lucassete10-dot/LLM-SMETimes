"""Prepare actual-plot contact sheets; record only explicitly inspected pages."""
import argparse
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

from PIL import Image,ImageDraw


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def prepare_role(folder):
    manifest=read(folder/'figure_manifest.json')
    figures=[]
    for row in manifest['figures']:
        figures.append({**row,'png':str(Path(row['png']).resolve()),'sha256':sha(row['png'])})
    review_path=folder/'visual_review.json'
    if review_path.exists():
        review=read(review_path)
        for row in review['figures']:
            assert sha(row['png'])==row['sha256'],'Reviewed source plot changed'
    else:
        review={'scope':'Actual original model prediction plots; all uninspected figures remain pending',
                'review_method':'Assistant viewed listed contact-sheet pages; source PNG SHA256 and identity bound',
                'figures':[],'inspected_pages':[],'notes':[],'numeric_success_is_separate':True}
    already={r['png'] for r in review['figures']}
    pending=[r for r in figures if r['png'] not in already]
    sheets=folder/'review_contact_sheets';sheets.mkdir(exist_ok=True)
    pages=[]
    for begin in range(0,len(figures),9):
        rows=figures[begin:begin+9]
        page=Image.new('RGB',(2100,1350),'white')
        draw=ImageDraw.Draw(page)
        for i,row in enumerate(rows):
            with Image.open(row['png']) as original:
                im=original.convert('RGB');im.thumbnail((700,420))
                page.paste(im,(i%3*700,i//3*450))
            draw.text((i%3*700+8,i//3*450+425),f'{row["cell_id"]} | {row["role"]}',fill='black')
        file=sheets/f'{begin//9+1:02d}.png';page.save(file)
        pages.append({'page':begin//9+1,'png':str(file.resolve()),'sha256':sha(file),'figures':rows})
    save(sheets/'index.json',pages)
    review['pending_figures']=pending
    save(review_path,review)
    manifest['visual_review']={'state':'pending' if not already else 'partial' if pending else 'completed',
                               'reviewed':len(already),'pending':len(pending),'review_file':str(review_path.resolve()),
                               'review_sha256':sha(review_path),'numeric_accuracy_pass_not_implied':True}
    save(folder/'figure_manifest.json',manifest)
    return {'folder':str(folder.resolve()),'figures':len(figures),'reviewed':len(already),'pages':len(pages)}


def mark(folder,page_numbers,notes):
    assert notes.strip(),'Actual visual findings required'
    pages=read(folder/'review_contact_sheets/index.json')
    selected=[p for p in pages if p['page'] in page_numbers]
    assert len(selected)==len(set(page_numbers)),'Requested page does not exist'
    review_path=folder/'visual_review.json';review=read(review_path)
    existing={r['png']:r for r in review['figures']}
    for page in selected:
        assert sha(page['png'])==page['sha256'],'Inspected contact sheet changed'
        for row in page['figures']:
            assert sha(row['png'])==row['sha256'],'Inspected source figure changed'
            existing[row['png']]=row
        review['inspected_pages'].append({'page':page['page'],'png':page['png'],'sha256':page['sha256'],
                                          'utc':datetime.now(timezone.utc).isoformat(),'notes':notes})
    review['figures']=list(existing.values())
    review['pending_figures']=[r for r in review['pending_figures'] if r['png'] not in existing]
    review['notes'].append(notes)
    review['utc']=datetime.now(timezone.utc).isoformat()
    save(review_path,review)
    manifest=read(folder/'figure_manifest.json')
    manifest['visual_review']={'state':'partial' if review['pending_figures'] else 'completed',
                               'reviewed':len(existing),'pending':len(review['pending_figures']),
                               'review_file':str(review_path.resolve()),'review_sha256':sha(review_path),
                               'numeric_accuracy_pass_not_implied':True}
    save(folder/'figure_manifest.json',manifest)
    trial=folder.parent/'trial_report_manifest.json'
    if trial.exists():
        record=read(trial)
        record['visual_review']='completed' if all(
            (folder.parent/role/'visual_review.json').exists() and not read(folder.parent/role/'visual_review.json')['pending_figures']
            for role in record['roles']) else 'partial'
        save(trial,record)
    return {'reviewed':len(existing),'pending':len(review['pending_figures'])}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--prepare',action='store_true');ap.add_argument('--task');ap.add_argument('--role',choices=['training','validation','test'])
    ap.add_argument('--pages',nargs='+',type=int);ap.add_argument('--notes')
    args=ap.parse_args()
    if args.prepare:
        q=read(args.root/'queue.json');summaries=[]
        for task in q['tasks']:
            if task['state']!='complete' or not task.get('train_validation_report_complete'):continue
            for role in ('training','validation'):
                folder=args.root/'reports'/task['key']/role
                summaries.append(prepare_role(folder))
        save(args.root/'visual_review_progress.json',{'roles':summaries,'all_generated_plots_reviewed':bool(summaries and all(r['reviewed']==r['figures'] for r in summaries))})
        print(json.dumps(summaries))
    else:
        assert args.task and args.role and args.pages and args.notes
        print(json.dumps(mark(args.root/'reports'/args.task/args.role,args.pages,args.notes)))


if __name__=='__main__':main()
