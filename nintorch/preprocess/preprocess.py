import numpy as np
from torch import Tensor

__all__ = ['MinMaxScalerNd']


class MinMaxScalerNd:
    def __init__(self, feature_range: tuple[float, float]) -> None:
        assert len(feature_range) == 2
        assert feature_range[0] < feature_range[1]
        self.feature_range = feature_range
        self.min = None
        self.max = None

    def fit(self, input: np.ndarray | Tensor) -> None:
        if isinstance(input, Tensor):
            min_ = input.min(dim=0)[0]
            max_ = input.max(dim=0)[0]
        elif isinstance(input, np.ndarray):
            min_ = np.min(input, axis=0)
            max_ = np.max(input, axis=0)
        else:
            raise TypeError(
                f'type of `input` or {type(input)} must be [ndarray, Tensor]'
            )
        self.min, self.max = min_, max_

    def transform(self, input: np.ndarray | Tensor) -> np.ndarray | Tensor:
        assert (
            self.min is not None and self.max is not None
        ), '`MinMaxScaleNd.fit` must be called first.'
        input = (input - self.min) / (self.max - self.min)
        input = (
            input * (self.feature_range[1] - self.feature_range[0])
            + self.feature_range[0]
        )
        return input

    def fit_transform(self, input: np.ndarray | Tensor) -> np.ndarray | Tensor:
        self.fit(input)
        return self.transform(input)
