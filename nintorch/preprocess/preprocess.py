import numpy as np

__all__ = ['MinMaxScaleNd']


class MinMaxScaleNd:
    def __init__(self, feature_range: tuple[float, float]) -> None:
        assert len(feature_range) == 2
        assert feature_range[0] < feature_range[1]
        self.feature_range = feature_range
        self.min = None
        self.max = None

    def fit(self, input: np.ndarray) -> None:
        min_ = np.min(input, axis=0)
        max_ = np.max(input, axis=0)
        self.min, self.max = min_, max_

    def transform(self, input: np.ndarray) -> np.ndarray:
        assert self.min is not None and self.max is not None
        input = (input - self.min) / (self.max - self.min)
        input = (
            input * (self.feature_range[1] - self.feature_range[0])
            + self.feature_range[0]
        )
        return input

    def fit_transform(self, input: np.ndarray) -> np.ndarray:
        self.fit(input)
        return self.transform(input)


if __name__ == '__main__':
    from sklearn.preprocessing import MinMaxScaler

    input = np.random.rand(100, 10)
    feature_range = (0, 1)
    custom = MinMaxScaleNd(feature_range)
    custom_output = custom.fit_transform(input)

    sklearn = MinMaxScaler(feature_range=feature_range)
    sklearn_output = sklearn.fit_transform(input)
    np.testing.assert_allclose(custom_output, sklearn_output)
