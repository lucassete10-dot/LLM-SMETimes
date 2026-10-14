"""Read-only raw-trace evidence for spikes; never excludes records by error."""
import json
from datetime import datetime, timezone
from pathlib import Path
import h5py
import numpy as np
from audit_mit_label_free_extraction_v2 import MIT_SOURCE
from prepare_mit import read_ref, discharge_slice
from scripts.audit_mit_discharge_integrity import discontinuity
from scripts.prepare_same_type_forecast import sha, save_json


def main():
    root = Path('experiments/mit_curve_spikes_v3') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True)
    print('RUN=' + str(root.resolve()), flush=True)
    source = MIT_SOURCE[1]
    save_json(root / 'config.json', dict(source=str(source), source_sha256=sha(source),
        script_sha256=sha(Path(__file__)), cells=['b1c0', 'b1c1', 'b1c2', 'b1c3'],
        purpose='raw integrity diagnosis only; no fitting/selection',
        spike_candidate_rule='Q below average immediate neighbors by >0.005 Ah, diagnostic only',
        automatic_exclusions=False, future_information_used_for_prediction=False))
    summaries = []
    with h5py.File(source, 'r') as mat:
        for index in range(4):
            cell = f'b1c{index}'
            summary = mat[mat['batch']['summary'][index, 0]]
            cycles = mat[mat['batch']['cycles'][index, 0]]
            numbers = np.asarray(summary['cycle']).reshape(-1).astype(int)
            q = np.asarray(summary['QDischarge']).reshape(-1)
            assert len(q) == len(numbers) == len(cycles['I'])
            neighbor_drop = np.zeros(len(q))
            neighbor_drop[1:-1] = (q[:-2] + q[2:]) / 2 - q[1:-1]
            rows = []
            for j, number in enumerate(numbers):
                t, i, v, qd = [read_ref(mat, cycles[key][j, 0]) for key in ('t', 'I', 'V', 'Qd')]
                reason, gap = discontinuity(t, i)
                row = dict(cycle=int(number), summary_capacity_ah=float(q[j]),
                    neighbor_drop_ah=float(neighbor_drop[j]),
                    spike_candidate=bool(neighbor_drop[j] > .005), integrity_reason=reason,
                    largest_adjacent_discharge_sample_gap_s=gap)
                valid = len(t) == len(i) == len(v) == len(qd) and np.isfinite(np.c_[t, i, v, qd]).all()
                run = discharge_slice(t, i, v) if valid else None
                if run is not None:
                    seconds = (t[run] - t[run[0]]) * 60
                    integral = float(np.trapezoid(-i[run], seconds) / 3600)
                    row.update(discharge_duration_s=float(seconds[-1]),
                        discharge_start_v=float(v[run[0]]), discharge_end_v=float(v[run[-1]]),
                        discharge_current_mean_a=float(i[run].mean()),
                        integrated_discharge_ah=integral,
                        trace_qd_max_ah=float(qd.max()),
                        integral_summary_difference_ah=float(integral-q[j]),
                        trace_summary_difference_ah=float(qd.max()-q[j]))
                else:
                    row['signal_issue'] = 'invalid_or_no_300s_main_discharge'
                rows.append(row)
            save_json(root / (cell + '_records.json'), rows)
            spikes = [r for r in rows if r['spike_candidate']]
            def avg(rs, key):
                values = [r[key] for r in rs if key in r]
                return float(np.median(values)) if values else None
            normal = [r for r in rows if not r['spike_candidate'] and r['cycle'] > 100]
            result = dict(cell_id=cell, cycles=len(rows), spike_candidates=len(spikes),
                future_spike_candidates=sum(r['cycle'] > 100 for r in spikes),
                physically_flagged_spike_candidates=sum(bool(r['integrity_reason'] or r.get('signal_issue')) for r in spikes),
                median_spike_end_v=avg(spikes, 'discharge_end_v'),
                median_other_future_end_v=avg(normal, 'discharge_end_v'),
                median_spike_integral_difference_ah=avg(spikes, 'integral_summary_difference_ah'),
                median_other_future_integral_difference_ah=avg(normal, 'integral_summary_difference_ah'),
                top_spikes=sorted(spikes, key=lambda r:r['neighbor_drop_ah'], reverse=True)[:12],
                exclusions_applied=0)
            summaries.append(result)
            print(json.dumps({k:v for k,v in result.items() if k != 'top_spikes'}), flush=True)
    save_json(root / 'summary.json', dict(cells=summaries, complete=True,
        limitation='A local capacity dip is not proof of invalid data. Trace integrity, cutoff and protocol require interpretation; no records were removed.',
        new_fits=0, goal_complete=False))


if __name__ == '__main__':
    main()
