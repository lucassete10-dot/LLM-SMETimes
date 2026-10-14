"""Copy current PNGs into the user-requested dataset/cell.png layout."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def main():
    root=Path('D:/LLM')/('SOH_current_test_figures_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
    root.mkdir(exist_ok=False)
    deliveries=Path.cwd()/'deliveries'
    groups={
        'MIT':deliveries/'MIT_main_20261008_five_seed/MIT/input_100',
        'Zhu2022_NCA':deliveries/'NCA_main_20261008T065806_427489Z/Zhu2022_NCA/input_100',
    }
    counts={};hashes={};figures=[]
    for dataset,source in groups.items():
        dest=root/dataset;dest.mkdir()
        paths=sorted(source.glob('*.png'));counts[dataset]=len(paths)
        assert len(paths)=={'MIT':107,'Zhu2022_NCA':44}[dataset]
        for path in paths:
            target=dest/path.name;shutil.copy2(path,target)
            relative=target.relative_to(root).as_posix()
            digest=hashlib.sha256(target.read_bytes()).hexdigest()
            assert digest==hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[relative]=digest
            figures.append(dict(dataset=dataset,cell_id=path.stem,figure=relative,
                input_cycles=100,evaluation_identity='held-out development cell',
                visual_review='not fully reviewed',source_png_sha256=digest))
    (root/'image_inventory.json').write_text(json.dumps(dict(png_count=151,
        counts_including_overviews=counts,overview_count=0,sha256=hashes),indent=2),encoding='utf-8')
    (root/'evaluation_manifest.json').write_text(json.dumps(dict(figures=figures,
        input_length_status='100 source cycles is provisional, not a verified optimum',
        missing_current_main_figures=['XJTU','HUST','UConn_ISU_ILCC'],
        historical_NASA_included=False,independent_test=False,goal_complete=False),indent=2),encoding='utf-8')
    (root/'README.txt').write_text('按数据集直接放置每颗电池的PNG。MIT107张，Zhu2022_NCA44张，共151张。\n'
        '与参考文件夹一样按数据集/电芯名.png组织，本包尚没有overview.png。\n'
        '只放当前已生成的新版本主图，不用旧版本补齐252张。XJTU、HUST、UConn新版本全量主图待完成。\n'
        '均为留出电芯开发验证，非独立测试。输入为100次源循环，尚非可靠最优解。部分曲线仍有明显预测偏差，完整人工审图未完成。\n',encoding='utf-8-sig')
    archive=root.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(root.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(root).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        names=z.namelist()
        assert sum(n.endswith('.png') for n in names)==151
        assert all(n.count('/')==1 for n in names if n.endswith('.png'))
    print(json.dumps(dict(folder=str(root),zip=str(archive),png_count=151,
        counts=counts,megabytes=round(archive.stat().st_size/1024**2,2),crc_passed=True)))


if __name__=='__main__':main()
