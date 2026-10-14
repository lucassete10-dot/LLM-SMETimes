"""Read-only CALCE Excel-to-signal adapter for protocol-matched Step 7 data.

The adapter reads no discharge-capacity column. It applies the audited Step 7
signal-only selection, duplicate removal, chronological segmentation, and
first-1000-second V/I feature extraction. Output has only cycle/raw_features.
"""

import argparse
from pathlib import Path

import numpy as np

from audit_calce_label_free_extraction_v2 import extract_folder, main as verify_training


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    adapt = sub.add_parser('adapt')
    adapt.add_argument('source_directory_or_xlsx', type=Path)
    adapt.add_argument('output_npz', type=Path)
    sub.add_parser('verify-training')
    args = parser.parse_args()
    if args.action == 'verify-training':
        verify_training()
        return
    output = args.output_npz.resolve()
    if output.suffix.lower() != '.npz' or output.exists():
        raise ValueError('output must be a new .npz file; existing data is never overwritten')
    if not output.parent.is_dir():
        raise ValueError('output directory must already exist')
    cycle, raw, detail = extract_folder(args.source_directory_or_xlsx)
    if len(cycle) <= 1:
        raise ValueError('CALCE inference requires at least two complete Step 7 diagnostics')
    np.savez_compressed(output, cycle=cycle, raw_features=raw)
    print('SAVED', output, len(cycle), 'diagnostics from', detail['workbooks'],
          'workbooks; capacity column not read', flush=True)


if __name__ == '__main__':
    main()
