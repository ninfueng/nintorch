import numpy as np
import torch

from nintorch.preprocess import MinMaxScalerNd


def test_minmax_scaler_nd():
    input = np.random.rand(100, 10)
    feature_range = (0, 1)
    scaler = MinMaxScalerNd(feature_range)
    np_out = scaler.fit_transform(input)

    input = torch.from_numpy(input)
    feature_range = (0, 1)
    scaler = MinMaxScalerNd(feature_range)
    torch_out = scaler.fit_transform(input)
    np.testing.assert_allclose(np_out, torch_out.numpy())
