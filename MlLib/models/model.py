import cupy as cp
from MlLib.losses import Loss, CategoricalCrossEntropy
from MlLib.optimizers import Optimizer, SGD
from timeit import default_timer as timer


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

    def backward(self, x: cp.ndarray) -> cp.ndarray:
        for layer in reversed(self.layers):
            x = layer.backward(x)
        return x

    def fit(self, x: cp.ndarray, y: cp.ndarray, epochs: int, verbose: int = 1):
        if not epochs or epochs < 1:
            raise ValueError("Epoch number must be greater than or equal to 1.")

        start = timer()

        if verbose >= 1:
            print(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            y_pred = self.forward(x)

            loss = self.loss(y, y_pred)

            if epochs + 1 < 10 or (epochs > 10 and epoch % 10 == 0):
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}")

            output = self.loss.backward(y, y_pred)

            self.backward(output)

            self.optimizer.update_params(self.layers)

        stop = timer()
        print(f"End training for {epochs} epochs in {stop - start:.2f} seconds")
