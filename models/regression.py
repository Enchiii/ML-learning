import timeit

import numpy as np

from losses import Loss


class LinearRegression:
    count = 0

    def __init__(self, loss: Loss, lr: float = 0.0001, name: str = None, _model: str = "LinearRegresion"):
        self.id = None
        self._model = _model
        if _model == "LinearRegresion":
            self.id = LinearRegression.count
            LinearRegression.count += 1

        self.name = name
        if name is None:
            self.name = f"{_model}_" + str(self.id)
        self.loss = loss
        self.lr = lr
        self.bias: float = 0.0
        self.weights: np.ndarray = np.array([])

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int, verbose: int = 1):

        if epochs <= 0:
            raise ValueError("epoch must be > 0")

        start = timeit.default_timer()

        if x.ndim == 1:
            x = np.expand_dims(x, axis=1)
        if y.ndim == 1:
            y = np.expand_dims(y, axis=1)
        m: int
        n: int
        m, n = x.shape
        self.bias = 0.0
        self.weights = np.zeros((n, 1))

        print(f"Fitting {self.name}")

        for epoch in range(epochs):
            y_pred: np.ndarray = self.predict(x)

            dw: np.ndarray = self._calc_weights(m, x, y, y_pred)
            db: float = self._calc_bias(m, y, y_pred)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if verbose > 0 and epoch % (epochs // 10) == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {self.loss(y, y_pred)}")

        stop = timeit.default_timer()
        print(f"\nFitted in {round(stop - start, 5)} sec, Loss: {round(self.loss(y, y_pred), 6)}")

    def _calc_weights(self, m, x, y, y_pred) -> np.ndarray:
        return (1 / m) * np.dot(x.T, (y_pred - y))

    def _calc_bias(self, m, y, y_pred) -> float:
        return (1 / m) * np.sum(y_pred - y)

    def predict(self, x: np.ndarray):
        if x.ndim == 1:
            x = np.expand_dims(x, axis=1)
        return np.dot(x, self.weights) + self.bias


class RidgeRegression(LinearRegression):
    count = 0

    def __init__(self, loss: Loss, lr: float = 0.0001, name: str = None, alpha: float = 1, _model: str = "RidgeRegression"):
        super().__init__(loss, lr, name, _model=_model)
        self.id = RidgeRegression.count
        RidgeRegression.count += 1
        self.alpha = alpha

    def _calc_weights(self, m, x, y, y_pred) -> np.ndarray:
        return (1 / m) * (np.dot(x.T, (y_pred - y)) + (self.alpha * self.weights))

    def _calc_bias(self, m, y, y_pred) -> float:
        return (1 / m) * np.sum(y_pred - y)


class LassoRegression(LinearRegression):
    count = 0

    def __init__(self, loss: Loss, lr: float = 0.0001, name: str = None, alpha: float = 1, _model: str = "LassoRegression"):
        super().__init__(loss, lr, name, _model=_model)
        self.id = RidgeRegression.count
        RidgeRegression.count += 1
        self.alpha = alpha

    def _calc_weights(self, m, x, y, y_pred) -> np.ndarray:
        return (1 / m) * np.dot(x.T, (y_pred - y)) + self.alpha * np.sign(self.weights)

    def _calc_bias(self, m, y, y_pred) -> float:
        return (1 / m) * np.sum(y_pred - y)
