"""Read-only OOXML verification against independently recomputed experiment JSON."""
import json
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
from scripts.prepare_same_type_forecast import sha, save_json

NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
      'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}


def main():
    output = Path('outputs/01a0c826-7cec-7363-ae70-be90675e6bf1')
    file = output / 'SOH四队列核验汇总_v2_20261007.xlsx'
    root = Path('results/goal_v2_tables/20261007T111732_433233Z')
    read = lambda name: json.loads((root / name).read_text(encoding='utf-8'))
    checked = 0
    with zipfile.ZipFile(file) as z:
        shared = []
        if 'xl/sharedStrings.xml' in z.namelist():
            shared = [''.join(si.itertext()) for si in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        relationships = {r.attrib['Id']: r.attrib['Target'] for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        data = {}
        for sh in ET.fromstring(z.read('xl/workbook.xml')).find('s:sheets', NS):
            target = relationships[sh.attrib['{' + NS['r'] + '}id']]
            target = target.lstrip('/') if target.startswith('/') else 'xl/' + target
            values = {}
            for cell in ET.fromstring(z.read(target)).findall('.//s:sheetData/s:row/s:c', NS):
                kind = cell.attrib.get('t')
                if kind == 'inlineStr':
                    value = ''.join(cell.find('s:is', NS).itertext())
                else:
                    node = cell.find('s:v', NS)
                    if node is None:
                        continue
                    text = node.text
                    assert kind != 'e', (sh.attrib['name'], cell.attrib['r'], text)
                    value = shared[int(text)] if kind == 's' else bool(int(text)) if kind == 'b' else float(text) if kind != 'str' else text
                values[cell.attrib['r']] = value
            data[sh.attrib['name']] = values
        assert len(data) == 8

        def compare(sheet, col, row, expected):
            nonlocal checked
            actual = data[sheet].get(col + str(row))
            if expected is None:
                assert actual is None
            elif isinstance(expected, (str, bool)):
                assert actual == expected, (sheet, col, row, actual, expected)
            else:
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
            checked += 1

        for i, r in enumerate(read('cohorts.json'), 7):
            for col, value in zip('ABCDEFGHIJKLMNO', [r['dataset'], r['cells'], r['summary']['mape_percent']['mean'],
                r['summary']['mape_percent']['seed_sd'], r['summary']['late_mape']['mean'], r['summary']['late_mape']['seed_sd'],
                r['summary']['rmse_soh']['mean'], r['summary']['rmse_soh']['seed_sd'], r['additional_metrics']['mae_soh']['mean'],
                r['additional_metrics']['r2']['mean'], r['a_numeric_pass'], r['mape_6_pass_count'], r['joint_pass_count'],
                r['short_coverage_cells'], r['selection_status']]):
                compare('四队列总览', col, i, value)
        fields = ['dataset', 'cell_id', 'fold', 'mape_percent', 'mape_percent_seed_sd', 'late_mape', 'late_mape_seed_sd',
                  'mae_soh', 'rmse_soh', 'r2', 'start_error_pp', 'endpoint_pp', 'drop_error_pp', 'truth_drop_pp',
                  'prediction_drop_pp', 'mape_6_pass', 'joint_pass', 'severe_flat_failure', 'failed_curve_criteria',
                  'short_coverage', 'future_points', 'future_first_cycle', 'future_last_cycle', 'future_span']
        for i, r in enumerate(read('per_cell.json'), 2):
            for col, field in zip('ABCDEFGHIJKLMNOPQRSTUVWX', fields):
                value = ', '.join(r[field]) if field == 'failed_curve_criteria' else r[field]
                # Empty strings may be omitted from OOXML without changing the cell.
                if value == '':
                    assert data['248电芯指标'].get(col + str(i)) in (None, '')
                else:
                    compare('248电芯指标', col, i, value)
        fields = ['dataset', 'cell_id', 'fold', 'seed', 'mape_percent', 'late_mape', 'mae_soh', 'rmse_soh', 'r2',
                  'start_error_pp', 'endpoint_pp', 'drop_error_pp', 'short_coverage', 'future_points', 'future_span']
        for i, r in enumerate(read('per_cell_seed.json'), 2):
            for col, field in zip('ABCDEFGHIJKLMNO', fields):
                compare('1240逐种子指标', col, i, r[field])
        for i, r in enumerate(read('point_prediction_sources.json'), 2):
            for col, value in zip('ABCDEF', [r['dataset'], r['fold'], r['seed'], r['file'], r['sha256'], ', '.join(r['cells'])]):
                compare('逐点预测来源', col, i, value)
            assert sha(r['file']) == r['sha256']
    save_json(output / 'goal_v2_workbook_ooxml_verification.json', dict(passed=True, checked_cells=checked,
              file_sha256=sha(file), table_cells=248, seed_records=1240,
              scope='saved OOXML values in summary, all cell/seed metrics and all NPZ links; no native Excel recalculation claim',
              source_sha256=sha(__file__), goal_complete=False))
    print('OOXML_PASS checked_cells=' + str(checked))


if __name__ == '__main__':
    main()
