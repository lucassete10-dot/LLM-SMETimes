"""One image-only archive for all currently exported dataset curves."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def main():
    project=Path.cwd().resolve()
    root=Path('D:/LLM')/('SOH_all_current_test_figures_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
    root.mkdir(exist_ok=False)
    groups={
        'MIT':(project/'deliveries/MIT_main_20261008_five_seed/MIT/input_100',107,5),
        'XJTU':(Path('D:/LLM/SOH_XJTU_test_figures_20261008T112553/XJTU'),44,1),
        'Zhu2022_NCA':(project/'deliveries/NCA_main_20261008T065806_427489Z/Zhu2022_NCA/input_100',44,5),
    }
    counts={};hashes={};figures=[]
    for dataset,(source,expected,seeds) in groups.items():
        dest=root/dataset;dest.mkdir()
        paths=sorted(source.glob('*.png'));assert len(paths)==expected
        counts[dataset]=len(paths)
        for source_file in paths:
            target=dest/source_file.name;shutil.copy2(source_file,target)
            digest=hashlib.sha256(target.read_bytes()).hexdigest()
            assert digest==hashlib.sha256(source_file.read_bytes()).hexdigest()
            relative=target.relative_to(root).as_posix();hashes[relative]=digest
            figures.append(dict(dataset=dataset,cell_id=target.stem,figure=relative,
                input_source_cycles=100,folds=5,seeds=seeds,
                evaluation_identity='held-out development cell',full_visual_review=False))
    pending=['HUST','UConn_ISU_ILCC']
    for dataset in pending:
        dest=root/dataset;dest.mkdir();counts[dataset]=0
        (dest/'尚未完成出图.txt').write_text('该数据集已开展实验，但当前阶段全量主曲线图尚未导出。\n此文件夹不代表图片或性能交付完成，后续补齐时更新同一个综合包。\n',encoding='utf-8-sig')
    total=sum(counts.values());assert total==195
    (root/'image_inventory.json').write_text(json.dumps(dict(png_count=total,
        counts_including_overviews=counts,overview_count=0,sha256=hashes),indent=2),encoding='utf-8')
    (root/'evaluation_manifest.json').write_text(json.dumps(dict(figures=figures,
        goal_datasets=['MIT','XJTU','HUST','UConn_ISU_ILCC','Zhu2022_NCA'],
        pending_main_curves=pending,independent_test=False,goal_complete=False,
        input_selection_status='100 source cycles provisional, not a reliably optimal N',
        shape_success_not_established=True),indent=2),encoding='utf-8')
    (root/'README.txt').write_text('五数据集当前主曲线综合包\n\n'
        'MIT：107张，五折×五种子。\nXJTU：44张，五折×一个种子。\nZhu2022_NCA：44张，五折×五种子。\n'
        'HUST、UConn–ISU–ILCC：当前阶段全量主图尚未导出，文件夹中明确标注待补齐。\n'
        '合计195张，按数据集/电芯名.png放置，无overview.png。\n\n'
        '所有图片输入前100次源循环，100尚不是已证明的可靠最优输入长度。\n'
        '为留出电芯开发验证，非独立测试。不删失败案例，不混入旧图；部分拐点和尾段仍明显偏离实际。\n'
        '图片经过自动字体及碰撞检查，完整人工审图尚未结束。\n'
        '当前目标不包含NASA，因此未加入历史NASA图片。\n',encoding='utf-8-sig')
    archive=root.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(root.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(root).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        names=z.namelist();assert sum(n.endswith('.png') for n in names)==195
        assert len(names)==len(set(names))
    print(json.dumps(dict(zip=str(archive),folder=str(root),images=total,counts=counts,
        megabytes=round(archive.stat().st_size/1024**2,2),crc_passed=True)))


if __name__=='__main__':main()
