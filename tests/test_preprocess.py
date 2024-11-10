import numpy as np

from nintorch.preprocess import AMinMaxScaler


def test_minmax_scaler_nd():
    input = np.random.rand(100, 10)
    feature_range = (0, 1)
    scaler = AMinMaxScaler(feature_range)
    out = scaler.fit_transform(input)
    assert np.amax(out) <= 1
    assert np.amin(out) >= 0
