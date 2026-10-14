"""Signal-inferred level with a frozen Ridge forecast shape."""
from battery_smetimes.research_baselines import predict_baseline
from battery_smetimes.nasa_cycle_calibration import predict as calibrated_predict


def predict(bundle,cell,future):
    p=predict_baseline(bundle['primary'],cell,100,future)
    if bundle['calibration'] is None:return p
    a=calibrated_predict(bundle['calibration'],cell,[101.])[0]
    b=predict_baseline(bundle['primary'],cell,100,[101.])[0]
    return p+(a-b)
