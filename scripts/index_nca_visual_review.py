"""Index completed human reviews; never infer human inspection from file existence."""
import json
from pathlib import Path
import re
from scripts.prepare_same_type_forecast import save_json, sha

ROOT = Path('results/zhu2022_nca_heldout_curves/20261004T163448_031653Z')


def main():
    manifest = json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
    reviews = []
    for line in (ROOT/'visual_review.md').read_text(encoding='utf-8').splitlines():
        match = re.match(r'\| (CY[^|]+) \| (Ridge|ExtraTrees) \| (.+) \|$', line)
        if match:
            cell, model, finding = match.groups()
            reviews.append(dict(cell_id=cell.strip(), model={'Ridge':'ridge', 'ExtraTrees':'extra_trees'}[model], finding=finding))
    indexed = {(r['cell_id'], r['model']): r for r in reviews}
    assert len(indexed) == len(reviews) == 110
    expected = {(r['cell_id'], r['model']) for r in manifest['figures']}
    assert set(indexed) == expected
    fields = {'MAPE':'mean_per_seed_mape', 'late':'mean_late_half_mape', 'endpoint':'mean_endpoint_soh_pp'}
    for figure in manifest['figures']:
        record = indexed[(figure['cell_id'], figure['model'])]
        png = Path(figure['figure']+'.png')
        assert sha(png) == figure['png_sha256']
        for label, field in fields.items():
            number = re.search(label+r'([0-9.]+)', record['finding'])
            if number:
                assert float(number[1]) == round(figure['metrics'][field], 2), record
        record.update(png_sha256=figure['png_sha256'], figure=str(png),
                      human_inspection_recorded=True, model_accuracy_approved=False)
    save_json(ROOT/'visual_review_index.json', dict(
        all110_human_review_records_present=True, reviews=reviews,
        metrics_transcription_checked=True, image_hashes_verified=True,
        original_export_manifest_retained=True,
        export_manifest_pending_superseded_by_this_review_index=True,
        human_review_source_sha256=sha(ROOT/'visual_review.md'),
        index_script_sha256=sha(__file__),
        caution='This indexes explicit human review notes; it does not independently perform visual inspection or approve model accuracy.'))
    print('INDEXED110 HUMAN REVIEW RECORDS; IMAGE HASHES AND TRANSCRIBED METRICS CHECKED')


if __name__ == '__main__':
    main()
