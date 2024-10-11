import numpy as np

from losses import MSE
from models.regression import LinearRegression


def get_y(x: int | float, a: int | float = 1, b: int | float = 0):
    return a * x + b


if __name__ == "__main__":
    N: int = 10
    x: np.ndarray = np.array([x for x in range(N)])
    y: np.ndarray = np.array([get_y(x, 1, 0) for x in x])

    model: LinearRegression = LinearRegression(loss=MSE(), lr=0.01)
    model.fit(x, y, epochs=100)
