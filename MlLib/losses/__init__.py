import numpy as np
import cupy as cp


class Loss:
    count = 0

    def __init__(self, name: str | None = None):
        self.id = Loss.count
        Loss.count += 1
        self.name = name
        self.type = None
        if name is None:
            self.name = f"Loss_" + str(self.id)

    def info(self) -> None:
        print(f"loss type: {self.type}")
        print(f"loss name: {self.name}")

    def __call__(self, y_true: np.ndarray | None = None, y_pred: np.ndarray | None = None, verbose: int = 0):
        if verbose:
            self.info()


class MSE(Loss):
    count = 0

    def __init__(self, name=None):
        super().__init__()
        self.id = MSE.count
        MSE.count += 1
        self.name = name
        self.type = "MSE"
        if name is None:
            self.name = f"{self.type}_" + str(self.id)

    def __call__(self, y_true: np.ndarray | None = None, y_pred: np.ndarray | None = None, verbose: int = 0) -> float:
        super().__call__(y_true, y_pred, verbose)
        return np.mean((y_true - y_pred) ** 2)


class RMSE(Loss):
    count = 0

    def __init__(self, name=None):
        super().__init__(name)
        self.id = RMSE.count
        RMSE.count += 1
        self.name = name
        self.type = "RMSE"
        if name is None:
            self.name = f"{self.type}_" + str(self.id)

    def __call__(self, y_true: np.ndarray | None = None, y_pred: np.ndarray | None = None, verbose: int = 0) -> float:
        super().__call__(y_true, y_pred, verbose)
        return np.sqrt(MSE()(y_true, y_pred))


class MAE(Loss):
    count = 0

    def __init__(self, name=None):
        super().__init__(name)
        self.id = MAE.count
        MAE.count += 1
        self.name = name
        self.type = "MAE"
        if name is None:
            self.name = f"{self.type}_" + str(self.id)

    def __call__(self, y_true: np.ndarray | None = None, y_pred: np.ndarray | None = None, verbose: int = 0) -> float:
        super().__call__(y_true, y_pred, verbose)
        return np.mean(np.abs(y_true - y_pred))


# classification loss functions
class CategoricalCrossEntropy(Loss):
    count = 0

    def __init__(self, name=None):
        super().__init__(name)
        self.id = CategoricalCrossEntropy.count
        CategoricalCrossEntropy.count += 1
        self.name = name
        self.type = "CategoricalCrossEntropy"
        if name is None:
            self.name = f"{self.type}_" + str(self.id)

    def __call__(
        self,
        y_true: np.ndarray | cp.ndarray | None = None,
        y_pred: np.ndarray | cp.ndarray | None = None,
        verbose: int = 0,
    ) -> float:
        super().__call__(y_true, y_pred, verbose)
        y_pred = cp.clip(y_pred, 1e-15, 1 - 1e-15)
        return -cp.sum(y_true * cp.log(y_pred), axis=-1)


class SparseCategoricalCrossEntropy(Loss):
    count = 0

    def __init__(self, name=None):
        super().__init__(name)
        self.id = SparseCategoricalCrossEntropy.count
        SparseCategoricalCrossEntropy.count += 1
        self.name = name
        self.type = "SparseCategoricalCrossEntropy"
        if name is None:
            self.name = f"{self.type}_" + str(self.id)

    def __call__(self, y_true: np.ndarray | None = None, y_pred: np.ndarray | None = None, verbose: int = 0) -> float:
        super().__call__(y_true, y_pred, verbose)
        pass
