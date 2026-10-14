"""Audit all70 frozen CV curve grids and replay their plotting pixels in memory.

The plotting helper was absent from the historical source snapshot. Its current
source is preserved explicitly as a replay dependency, NOT a recovered snapshot.
"""
import ast
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil
import traceback
import joblib
import numpy as np
from PIL import Image
from scripts.prepare_same_type_forecast import save_json, sha

RUN = Path('experiments/stm32_compact_trees/runs/20261004T074429_087457Z')
HELPER = Path('scripts/train_soh_research_v5.py')


class MemoryFolder:
    def __init__(self):
        self.buffer = io.BytesIO()

    def __truediv__(self, name):
        assert name == 'curves_origin100.png'
        return self.buffer


def plotting_function():
    tree = ast.parse(HELPER.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'save_scores')
    # Suppress archive writes only; the original plotting statements are intact.
    remove = {'folder.mkdir', 'save_json', 'np.savez_compressed'}
    node.body = [n for n in node.body if not
                 (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                  and ast.unparse(n.value.func) in remove)]
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {'np': np}
    exec(compile(module, str(HELPER), 'exec'), namespace)
    return namespace['save_scores']


def main():
    out = Path('experiments/heldout_curve_archive_audit')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True)
    shutil.copy2(__file__, out/Path(__file__).name)
    shutil.copy2(HELPER, out/'replay_plotting_dependency.py')
    rows = []
    try:
        cfg = json.loads((RUN/'config.json').read_text())
        plans = json.loads((RUN/'splits.json').read_text())['plans']
        assert plans == json.loads((RUN/'frozen_audit/splits.json').read_text())['plans']
        src = RUN/'source_snapshot/scripts/train_stm32_compact_trees.py'
        assert sha(src) == next(v for k, v in cfg['source_sha256'].items() if k.endswith('train_stm32_compact_trees.py'))
        source = src.read_text(encoding='utf-8')
        assert "evaluate_baseline(bundle,cells,plan['test'],predictor=predict);save_scores(dest/'test',report,arrays)" in source
        log = (RUN/'training.log').read_text(encoding='utf-8')
        plot = plotting_function()
        for plan in plans:
            dev = set(plan['train']) | set(plan['validation'])
            test = set(plan['test'])
            assert not dev & test
            for seed in cfg['seeds']:
                folder = RUN/plan['name']/str(seed)
                assert f"SAVED {plan['name']} seed={seed} " in log
                model = joblib.load(folder/'model.joblib')
                assert set(model['train_ids']) == dev
                report = json.loads((folder/'test/metrics.json').read_text())
                assert not report['skipped']
                assert len(report['by_cell']) == len(test)
                assert {r['cell_id'] for r in report['by_cell']} == test
                array_path = folder/'test/predictions.npz'
                with np.load(array_path) as saved:
                    assert set(saved.files) == {k+'__100__'+f for k in test for f in ('cycle','truth','prediction')}
                    arrays = {k: saved[k].copy() for k in saved.files}
                memory = MemoryFolder()
                plot(memory, report, arrays)
                memory.buffer.seek(0)
                png = folder/'test/curves_origin100.png'
                with Image.open(png) as original, Image.open(memory.buffer) as replay:
                    a = np.asarray(original.convert('RGBA'))
                    b = np.asarray(replay.convert('RGBA'))
                    assert a.shape == b.shape
                    equal = bool(np.array_equal(a, b))
                    delta = int(np.max(np.abs(a.astype(np.int16)-b.astype(np.int16))))
                    assert equal, f'Plotting replay pixel mismatch: {png}, delta={delta}'
                    size = list(original.size)
                rows.append(dict(dataset=plan['dataset'], fold=plan['name'], seed=seed,
                    test_ids=plan['test'], panel_count=len(test), image_size=size,
                    model_sha256=sha(folder/'model.joblib'), predictions_sha256=sha(array_path),
                    image_sha256=sha(png), pixels_identical_to_saved_outer_predictions=equal))
                print(f"VERIFIED {plan['name']} seed={seed} panels={len(test)} pixels_equal={equal}", flush=True)
        assert len(rows) == 70
        save_json(out/'verification.json', dict(passed=True, units=rows,
            total_grids=len(rows), total_cell_seed_panels=sum(r['panel_count'] for r in rows),
            historical_plot_helper_snapshot_missing=True,
            replay_dependency_sha256=sha(HELPER),
            replay_dependency_role='Current source captured for exact pixel replay; not a historical snapshot',
            rendering_role='Memory-only archive provenance verification, no new deliverable figures',
            training_log_sha256=sha(RUN/'training.log'), run=str(RUN.resolve())))
        print('OUTPUT='+str(out.resolve()), flush=True)
    except BaseException:
        save_json(out/'failure.json', dict(traceback=traceback.format_exc(), checked_units=rows))
        raise


if __name__ == '__main__':
    main()
