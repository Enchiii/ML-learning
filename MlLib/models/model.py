import cupy as cp
from MlLib.losses import Loss, CategoricalCrossEntropy
from MlLib.optimizers import Optimizer, SGD
from timeit import default_timer as timer


class LossActivation:
    def __init__(self, loss: Loss, activation):
        self.activation = activation
        self.loss = loss

    def backward(self, dvalues, y_true):
        pass


class Model:
    def __init__(self, layers: list, loss: Loss, optimizer: Optimizer):
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer  # zamiast learning rate dajemy optimizergdzie ustawimy lr

    def add_layer(self, layer):
        self.layers.append(layer)

    def predict(self, x: cp.ndarray) -> cp.ndarray:
        return self.forward(x)

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def fit(self, x: cp.ndarray, y: cp.ndarray, epochs: int, verbose: int = 1):
        if not epochs or epochs < 1:
            raise ValueError("Epoch number must be greater than or equal to 1.")

        start = timer()

        if verbose >= 1:
            print(f"Starting training for {epochs} epochs...")


        for epoch in range(epochs):
            y_pred = self.forward(x)
