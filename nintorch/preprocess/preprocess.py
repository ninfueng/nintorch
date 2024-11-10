import numpy as np
from torch import Tensor

__all__ = ['AMinMaxScaler']


class AMinMaxScaler:
    def __init__(self, feature_range: tuple[float, float]) -> None:
        assert len(feature_range) == 2
        assert feature_range[0] < feature_range[1]
        self.feature_range = feature_range
        self.amin = None
        self.amax = None

    def fit(self, input: np.ndarray) -> None:
        amin = np.amin(input)
        amax = np.amax(input)
        self.amin, self.amax = amin, amax

    def transform(self, input: np.ndarray) -> np.ndarray:
        assert self.amin is not None
        assert self.amax is not None
        input = (input - self.amin) / (self.amax - self.amin)
        input = (
            input * (self.feature_range[1] - self.feature_range[0])
            + self.feature_range[0]
        )
        return input

    def fit_transform(self, input: np.ndarray | Tensor) -> np.ndarray | Tensor:
        self.fit(input)
        return self.transform(input)


if __name__ == '__main__':
    input = np.random.rand(1, 10)
    print(input)

    scaler = AMinMaxScaler(feature_range=(-1, 1))
    output = scaler.fit_transform(input)
    print(output)
