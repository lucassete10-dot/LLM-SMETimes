"""Same inferred101 anchor; blend independent training-derived future shapes."""
import numpy as np
from battery_smetimes.nasa_cycle_calibration import predict as calibrated
from battery_smetimes.research_baselines import predict_baseline


def predict(primary,calibration,query,future,weight):
    weight=float(weight)
    if not np.isfinite(weight) or not 0<=weight<=1:raise ValueError('invalid weight')
    direct=calibrated(calibration,query,future)
    base=predict_baseline(primary,query,100,future)
    level=base+calibrated(calibration,query,[101.])[0]-predict_baseline(primary,query,100,[101.])[0]
    return (1-weight)*level+weight*direct
