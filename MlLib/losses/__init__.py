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

    def forward(self, y_true, y_pred):
        pass

    def backward(self, y_true: cp.ndarray, dvalues: cp.ndarray) -> cp.ndarray:
        pass


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

        self.dinputs: cp.ndarray | None = None

    def __call__(
        self,
        y_true: cp.ndarray | None = None,
        y_pred: cp.ndarray | None = None,
        verbose: int = 0,
    ):
        super().__call__(y_true, y_pred, verbose)

        return cp.mean(self.forward(y_true, y_pred))

    def forward(self, y_true: cp.ndarray, y_pred: cp.ndarray) -> cp.ndarray:
        y_pred_clipped = cp.clip(y_pred, 1e-7, 1 - 1e-7)
        correct_confidences = cp.sum(y_pred_clipped * y_true, axis=1)
        return -cp.log(correct_confidences)

    def backward(self, y_true: cp.ndarray, dvalues: cp.ndarray) -> cp.ndarray:
        # samples = dvalues.shape[0]
        # self.dinputs = -y_true / dvalues
        # # Normalize gradient
        # self.dinputs = self.dinputs / samples
        # return self.dinputs
        # Number of samples
        samples = len(dvalues)
        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = cp.argmax(y_true, axis=1)
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples
        return self.dinputs


class BinaryCrossEntropy(Loss):
    def __call__(
        self,
        y_true: np.ndarray | cp.ndarray | None = None,
        y_pred: np.ndarray | cp.ndarray | None = None,
        verbose: int = 0,
    ) -> float:
        super().__call__(y_true, y_pred, verbose)
        y_pred = cp.clip(y_pred, 1e-15, 1 - 1e-15)
        return -cp.mean(y_true * cp.log(y_pred) + (1 - y_true) * cp.log(1 - y_pred))

    def backward(self, error, y_true):
        samples = y_true.shape[0]
        return (error - y_true) / samples


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
