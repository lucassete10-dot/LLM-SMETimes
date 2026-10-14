"""Carry visual reviews only across identical PNG bytes and figure identities."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(row):
    return row['cell_id'], row['fold'], row['role'], tuple(sorted(row['seeds']))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--destination', type=Path, required=True)
    args = ap.parse_args()
    plans = []
    # Validate everything before writing; never replace an existing review.
    for role in ('training', 'validation', 'test'):
        dest = args.destination / role
        if (dest / 'visual_review.json').exists():
            raise ValueError('Review already exists: ' + str(dest))
        manifest = read(dest / 'figure_manifest.json')
        source = args.source / role / 'visual_review.json'
        reviewed = {}
        notes = []
        if source.exists():
            src = read(source)
            notes.append({'source': str(source.resolve()), 'sha256': sha(source), 'notes': src['notes']})
            for row in src['figures']:
                if row['sha256'] != sha(Path(row['png'])):
                    raise ValueError('Previously reviewed image changed')
                key = identity(row)
                if key in reviewed:
                    raise ValueError('Duplicate source identity')
                reviewed[key] = row
        inherited, pending = [], []
        for row in manifest['figures']:
            png = Path(row['png']).resolve()
            digest = sha(png)
            old = reviewed.get(identity(row))
            if old and digest == old['sha256']:
                inherited.append({**old, 'png': str(png), 'inherited_from_review': str(source.resolve()),
                                  'source_review_sha256': sha(source), 'source_png': old['png']})
            else:
                pending.append({**row, 'png': str(png), 'sha256': digest})
        plans.append((dest, manifest, inherited, pending, notes))
    result = {}
    for dest, manifest, inherited, pending, notes in plans:
        sheet_dir = dest / 'pending_contact_sheets'
        sheet_dir.mkdir(exist_ok=True)
        pages = []
        for start in range(0, len(pending), 9):
            page = Image.new('RGB', (2100, 1350), 'white')
            rows = pending[start:start + 9]
            for index, row in enumerate(rows):
                with Image.open(row['png']) as original:
                    im = original.convert('RGB')
                    im.thumbnail((700, 450))
                    page.paste(im, (index % 3 * 700, index // 3 * 450))
            path = sheet_dir / f'{start // 9 + 1:02d}.png'
            page.save(path)
            pages.append({'png': str(path.resolve()), 'figures': rows})
        save(sheet_dir / 'index.json', pages)
        review_path = dest / 'visual_review.json'
        review = {'utc': datetime.now(timezone.utc).isoformat(),
                  'scope': 'Identical PNG figures inherited from actual prior visual inspection; remaining panels pending',
                  'review_method': 'Exact SHA256 plus cell, fold, role, and seeds equality',
                  'figures': inherited, 'pending_figures': pending, 'source_role_notes': notes,
                  'notes': 'Inherited findings apply only to the matched source figures; unmatched figures have not been visually inspected.',
                  'numeric_success_is_separate': True, 'goal_complete': False}
        save(review_path, review)
        manifest['visual_review'] = {'state': 'partial' if pending else 'completed',
                                     'review_file': str(review_path.resolve()), 'review_sha256': sha(review_path),
                                     'reviewed': len(inherited), 'pending': len(pending),
                                     'numeric_accuracy_pass_not_implied': True}
        save(dest / 'figure_manifest.json', manifest)
        result[dest.name] = {'inherited': len(inherited), 'pending': len(pending)}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
