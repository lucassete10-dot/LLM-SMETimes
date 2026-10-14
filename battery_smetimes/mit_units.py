"""Versioned conversion of author-normalized MIT current to physical amperes.

Author cell_analysis.m stores I = recorded Current / 1.1. The fixed 1.1 A
nominal is a published protocol constant, never an inferred cell capacity.
Legacy prepared caches are immutable and intentionally remain replayable.
"""
import numpy as np
from prepare_mit import ENHANCED_NAMES

MIT_NOMINAL_AMPERES_PER_C = 1.1
CURRENT_DERIVED_COLUMNS = tuple(i for i,name in enumerate(ENHANCED_NAMES)
    if name.startswith('current_') or name in ('prefix_charge_ah','prefix_energy_wh'))


def mit_prefix_to_amperes(raw):
    value=np.asarray(raw)
    if value.ndim!=2 or value.shape[1]!=len(ENHANCED_NAMES):raise ValueError('MIT enhanced signal schema mismatch')
    converted=value.copy()
    converted[:,CURRENT_DERIVED_COLUMNS]*=MIT_NOMINAL_AMPERES_PER_C
    return converted
