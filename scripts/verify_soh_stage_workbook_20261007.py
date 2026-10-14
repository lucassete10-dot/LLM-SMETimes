"""Read-only OOXML audit of exported static research-result workbook."""
import json,math,hashlib,zipfile,posixpath
from pathlib import Path
from xml.etree import ElementTree as ET
from statistics import mean

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/01a0c826-7cec-7363-ae70-be90675e6bf1'
AUDIT=ROOT/'experiments/four_cohort_progress_audit/20261006T152412_391404Z'
NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def equal(a,b):
    if b is None or isinstance(b,str) and b=='':assert a is None or a=='';return
    if isinstance(b,bool):assert type(a) is bool and a==b;return
    if isinstance(b,(int,float)):
        assert isinstance(a,(int,float)) and not isinstance(a,bool) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-10),(a,b)
    else:assert a==b,(a,b)


def main():
    book=OUT/'SOH阶段评测汇总_20261007.xlsx';proof=json.loads((OUT/'updated_workbook_verification.json').read_text())
    cohort=json.loads((AUDIT/'cohort_results.json').read_text());cells=json.loads((AUDIT/'cell_results.json').read_text())
    collected=json.loads((OUT/'collected_results.json').read_text());assert not collected['errors']
    with zipfile.ZipFile(book) as z:
        assert z.testzip() is None
        wb=ET.fromstring(z.read('xl/workbook.xml'));rels=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        mapping={r.attrib['Id']:r.attrib['Target'] for r in rels}
        strings=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            strings=[''.join(t.text or '' for t in s.findall('.//s:t',NS)) for s in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',NS)]
        tabs={};errors=[];tables=0;panes=0
        for sh in wb.findall('s:sheets/s:sheet',NS):
            target=mapping[sh.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']]
            path=target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/'+target)
            doc=ET.fromstring(z.read(path));values={}
            for cell in doc.findall('s:sheetData/s:row/s:c',NS):
                v=cell.find('s:v',NS);text=v.text if v is not None else None;typ=cell.get('t')
                if typ=='s':value=strings[int(text)]
                elif typ=='b':value=text=='1'
                elif typ=='inlineStr':value=''.join(t.text or '' for t in cell.findall('.//s:t',NS))
                elif typ in ('str','e'):value=text
                else:value=float(text) if text is not None else None
                values[cell.attrib['r']]=value
                if typ=='e':errors.append((sh.attrib['name'],cell.attrib['r'],text))
                assert cell.find('s:f',NS) is None,'Snapshot unexpectedly contains formulas'
            tables+=len(doc.findall('s:tableParts/s:tablePart',NS));panes+=len(doc.findall('s:sheetViews/s:sheetView/s:pane',NS))
            tabs[sh.attrib['name']]=values
        assert len(tabs)==11 and not errors and tables==11 and panes==11
        def checkrow(sheet,row,values):
            for i,value in enumerate(values):
                col='';n=i+1
                while n:n,rem=divmod(n-1,26);col=chr(65+rem)+col
                equal(tabs[sheet].get(col+str(row)),value)
        for n,r in enumerate(cohort,7):
            s=r['summary'];checkrow('当前完整评测',n,[r['dataset'],s['mape_percent']['mean'],s['mape_percent']['seed_sd'],s['late_mape']['mean'],s['rmse_soh']['mean'],r['additional_metrics']['mae_soh']['mean'],r['additional_metrics']['r2']['mean'],s['cells'],r['joint_cells'],s['joint_pass_fraction'],s['endpoint_pp']['mean'],s['drop_error_pp']['mean'],r['passed_gates'],', '.join(r['failing_gates']), '保留模型，未全部达标' if r['currently_retained'] else '完整开发CV候选，非最终采纳'])
        n=2;seedrow=2
        for block in cells:
            for r in block['by_cell']:
                rr=[a for a in block['rows'] if a['cell_id']==r['cell_id']];assert len(rr)==5
                def avg(f):
                    vv=[a[f] for a in rr if isinstance(a[f],(int,float)) and not isinstance(a[f],bool)]
                    return mean(vv) if vv else None
                fail=[name for condition,name in [(r['mape_percent']>6,'全MAPE'),(r['late_mape']>8,'后半段MAPE'),(r['endpoint_pp']>8,'末点'),(r['drop_error_pp']>5,'下降幅度')] if condition]
                checkrow('当前248电芯',n,[block['dataset'],r['cell_id'],r['mape_percent'],r['late_mape'],avg('mae_soh'),r['rmse_soh'],avg('r2'),r['endpoint_pp'],r['drop_error_pp'],r['truth_drop_pp'],r['prediction_drop_pp'],r['joint_pass'],', '.join(fail),any(a['short_coverage'] for a in rr),rr[0]['future_points'],rr[0]['future_span']]);n+=1
            for r in block['rows']:
                checkrow('当前逐种子指标',seedrow,[block['dataset'],r['fold'],r['cell_id'],r['seed'],r['mape_percent'],r['late_mape'],r['mae_soh'],r['rmse_soh'],r['r2'],r['endpoint_pp'],r['drop_error_pp'],r['truth_drop_pp'],r['prediction_drop_pp'],r['short_coverage'],r['future_points'],r['future_span']]);seedrow+=1
        assert n==250 and seedrow==1242
        for n,row in enumerate(collected['current'],7):checkrow('原保留基线',n,row)
        pilotrow=2
        for run in ['nca_segment_smoke/20261006T151635_589623Z','nca_quantile_smoke/20261006T153315_133191Z','nca_level_delta_smoke/20261006T153920_733893Z','nca_parameter_shape_smoke/20261007T045735_449912Z']:
            selection=json.loads((ROOT/'experiments'/run/'selection.json').read_text())
            assert json.loads((ROOT/'experiments'/run/'independent_verification.json').read_text())['passed']
            for r in selection['candidates']:
                p=r.get('parameters')
                if p is None:p={'quantile':r['quantile']} if r.get('quantile') is not None else {'tail_weight':r['tail_weight']} if r.get('tail_weight') is not None else {'control':True}
                checkrow('近期NCA小试',pilotrow,[run,r['candidate'],json.dumps(p,separators=(',',':')),r['mape_percent'],r['late_mape'],r['endpoint_pp'],r['drop_error_pp'],r['start_error_pp'],r['score'],'符合小试扩展条件' if selection['expand'] else '不扩展，保留原模型','1折1种子内层验证，非最终准确率']);pilotrow+=1
        assert pilotrow==19
        nasarow=2
        for run in ['nasa_cycle_calibration/20261005T154730_956753Z','nasa_calibrated_level/20261005T155241_501407Z','nasa_calibrated_level_formal/20261005T155706_400345Z']:
            records=json.loads((ROOT/'experiments'/run/'comprehensive_curve_metrics.json').read_text())
            assert records['count']==len(records['rows'])
            for r in records['rows']:
                checkrow('NASA历史完整指标',nasarow,[run,r['fold'],r['cell_id'],r['role'],r['candidate'],r['seed'],r['mae_soh'],r['rmse_soh'],r['mape_percent'],r['r2'],r['points'],r['future_observation_span'],r['short_coverage'],r['source_sha256']]);nasarow+=1
        assert nasarow==394
        for sheet,key in [('历史汇总','formal'),('候选阶段汇总','metric_files'),('逐芯结果','cell_details'),('运行与失败','runs'),('来源索引','inventory')]:
            for n,row in enumerate(collected[key],2):checkrow(sheet,n,row)
            expected=len(collected[key])+(4 if sheet=='来源索引' else 0);assert proof['rowCounts'][sheet]==expected
        # Added current evidence retains exact source digest in the final source index.
        for i,name in enumerate(['cohort_results.json','cell_results.json','config.json','source_evidence.json']):
            row=2+len(collected['inventory'])+i
            equal(tabs['来源索引']['E'+str(row)],hashlib.sha256((AUDIT/name).read_bytes()).hexdigest())
    result=dict(passed=True,workbook_sha256=hashlib.sha256(book.read_bytes()).hexdigest(),sheets=11,
        current_cells=248,current_cell_seed_rows=1240,all_historical_rows_compared=True,table_filters=11,freeze_panes=11,
        visual_review='All 11 sheet opening ranges reviewed; numeric export compared to sources.',
        engine_limit='OOXML/static-value export verified; Microsoft Excel application was not launched.',goal_complete=False)
    (OUT/'stage_workbook_independent_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
