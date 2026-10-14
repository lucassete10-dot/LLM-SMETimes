"""Package existing SOH development results on explicit user request."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    project=Path.cwd().resolve()
    out=project/'deliveries'/('SOH_results_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'.zip')
    sources={}
    def add(path,archive):
        path=Path(path)
        assert path.is_file(),path
        assert archive not in sources,archive
        sources[archive]=path
    folders=[('MIT',project/'deliveries/MIT_main_20261008_five_seed',107),
        ('Zhu2022_NCA',project/'deliveries/NCA_main_20261008T065806_427489Z',44)]
    for dataset,folder,count in folders:
        assert len(list(folder.rglob('*.png')))==count
        for path in folder.rglob('*'):
            if path.is_file() and (path.suffix in ('.png','.csv','.html','.md') or path.name=='manifest.json' or path.name.endswith('.source.json')):
                add(path,'curves/'+dataset+'/'+path.relative_to(folder).as_posix())
    groups={
        'MIT':[project/Path(s['run']) for s in json.loads((project/'experiments/core_length_cv/MIT_five_seed_summary_20261008.json').read_text())['sources']],
        'XJTU':[project/'experiments/core_length_cv/20261008T083003_121461Z'],
        'Zhu2022_NCA':[project/'experiments/nca_full_label_cv/20261008T065806_427489Z'],
        'UConn_ISU_ILCC':[project/'experiments/uconn_sparse_probe/20261008T073624_949393Z'],
        'HUST':[Path(r['run']) for r in json.loads((project/'experiments/hust_voltage_intervals_batches/20261008T074921_087742Z/runs.json').read_text())]
    }
    for dataset,runs in groups.items():
        for run in runs:
            proof=json.loads((run/'verification.json').read_text())
            assert proof['passed'],run
            prefix='numeric_results/'+dataset+'/'+run.name+'/'
            for name in ['config.json','verification.json','progress.json','coverage.json','all_rows.json','length_summary.json','training.log']:
                if (run/name).exists():add(run/name,prefix+name)
            for path in run.rglob('predictions.npz'):
                add(path,prefix+path.relative_to(run).as_posix())
            for path in run.rglob('metrics.json'):
                add(path,prefix+path.relative_to(run).as_posix())
    add(project/'experiments/core_length_cv/MIT_five_seed_summary_20261008.json','numeric_results/MIT/five_seed_summary.json')
    readme='''# 当前SOH预测结果（开发验证阶段）

图片：MIT 107张，Zhu2022_NCA 44张，共151张PNG。真实与预测都保留，包含失败电芯。MIT目录的index.html可打开逐张查看。

五个数据集的已核验数值结果、不同输入长度、逐点predictions.npz和配置分别位于numeric_results。XJTU、HUST、UConn尚无本阶段全量主图，不能把这个包称为五数据集完整交付。

MIT与NCA主图输入为100次源循环；100并非已证实最优输入长度。其他候选输入长度见配置和结果。MIT主配置为5折×5种子107颗；NCA为5折×5种子44颗。XJTU、HUST主要为单种子探索；UConn为单折单种子试验且容量参考仍待进一步核验。

所有结果为开发验证，非独立测试。低MAPE不等于曲线形状达标。图片阴影是种子SD，不是预测置信区间。没有用绘图平滑掩盖误差。

图片通过自动字体及碰撞检查，尚未完成全部逐张人工审查。表格CSV可用Excel打开；完整Excel交付尚未完成。模型权重与原始电池数据未收入本包。

本包不混入早期条件试验旧图或刚运行的未独立复核候选。manifest.json列出各文件SHA256。包已进行ZIP CRC完整性检查。
'''
    manifest=dict(images=151,image_counts={'MIT':107,'Zhu2022_NCA':44},
        datasets_with_numeric_results=list(groups),independent_test=False,goal_complete=False,
        full_visual_review=False,missing_main_curves=['XJTU','HUST','UConn_ISU_ILCC'],
        files=[dict(path=arc,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for arc,p in sorted(sources.items())])
    with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.writestr('README.md',readme)
        z.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        for arc,p in sorted(sources.items()):z.write(p,arc)
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
        names=z.namelist();assert sum(n.endswith('.png') for n in names)==151
        assert len(names)==len(set(names))
    print(json.dumps(dict(path=str(out),megabytes=round(out.stat().st_size/1024**2,2),images=151,files=len(names),crc_passed=True),ensure_ascii=False))


if __name__=='__main__':main()
