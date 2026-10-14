"""Reuse frozen CV loop with explicit full-label loader and separate output namespace."""
from pathlib import Path
import shutil
from scripts import train_nca_length_cv as runner
from scripts import nca_full_label_loader
from scripts.prepare_same_type_forecast import sha

INDEX=Path('data/interim/nca_early_label_recovery/20261008T065706_034115Z/index.json')


def main():
    original_path=runner.Path;original_save=runner.save_json
    def namespace(value):
        if str(value)=='experiments/nca_length_cv':value='experiments/nca_full_label_cv'
        return original_path(value)
    def record(path,data):
        if Path(path).name=='config.json' and 'plans' in data:
            data=dict(data,label_index=str(INDEX.resolve()),label_index_sha256=sha(INDEX),full_early_labels_loaded=True,
                limitation='Full recovered labels; previously examined development cells only. Actual short-input predictions now evaluated on observed N+future labels.')
            for file in (Path(__file__),Path('scripts/nca_full_label_loader.py')):
                data['source_sha256'][str(file)]=sha(file)
                shutil.copy2(file,Path(path).parent/'source_snapshot'/file.name)
        original_save(path,data)
    runner.Path=namespace
    runner.save_json=record
    runner.load_nominal=lambda plan:nca_full_label_loader.load(plan,INDEX)
    runner.main()


if __name__=='__main__':main()
