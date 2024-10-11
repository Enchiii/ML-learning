import timeit

import numpy as np

from losses import Loss


class LinearRegression:
    count = 0

    def __init__(self, loss: Loss, lr: float = 0.0001, name: str = None):
        self.id = LinearRegression.count
        LinearRegression.count += 1
        self.name = name
        if name is None:
            self.name = f"LinearRegression_" + str(self.id)
        self.loss = loss
        self.lr = lr
        self.bias: float = 0.0
        self.weights: np.ndarray = np.array([])

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int, verbose: int = 1):
        start = timeit.default_timer()

        if x.ndim == 1:
            x = np.expand_dims(x, axis=1)
        if y.ndim == 1:
            y = np.expand_dims(y, axis=1)
        m, n = x.shape
        self.bias = 0.0
        self.weights = np.zeros((n, 1))

        print(f"Fitting {self.name}")

        for epoch in range(epochs):
            y_pred: np.ndarray = self.predict(x)

            dw: float = (1 / m) * np.dot(x.T, (y_pred - y))
            db: float = (1 / m) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if verbose > 0 and epoch % (epochs // 10) == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {self.loss(y, y_pred)}")

        stop = timeit.default_timer()
        print(f"\nFitted in {round(stop - start, 5)} sec, Loss: {round(self.loss(y, y_pred), 6)}")

    def predict(self, x: np.ndarray):
        if x.ndim == 1:
            x = np.expand_dims(x, axis=1)
        return np.dot(x, self.weights) + self.bias
