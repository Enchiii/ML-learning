import cupy as cp

from MlLib.losses import Loss
from timeit import default_timer as timer


class Model:
    def __init__(self, layers: list | tuple, loss: Loss, learning_rate: float = 0.001):
        self.layers = layers
        self.loss = loss
        self.learning_rate = learning_rate

    def __call__(self, x: cp.ndarray) -> cp.ndarray:
        return self.forward(x)

    def predict(self, x: cp.ndarray) -> cp.ndarray:
        return self.forward(x)

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss) -> None:
        for layer in reversed(self.layers):
            loss = layer.backward(loss, self.learning_rate)

    def fit(self, x: cp.ndarray, y: cp.ndarray, epochs: int, learning_rate: float = None, verbose: int = 1) -> float:
        if not epochs or epochs < 1:
            raise ValueError("Epoch number must be greater than or equal to 1.")

        self.learning_rate = learning_rate if learning_rate else self.learning_rate

        if not self.learning_rate or self.learning_rate < 0:
            raise ValueError("Learning rate must be greater than 0.")

        start = timer()

        if verbose >= 1:
            print(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            y_pred = self.forward(x)
            loss = self.loss(y, y_pred)

            self.backward(self.loss.backward(loss, y))

            avg_loss = loss.mean()
            if verbose == 1:
                if (epoch + 1) % (epochs // 10) == 0:
                    print(f"Epoch {epoch+1}/{epochs} {(epoch+1)/epochs * 100:.1f}%\tLoss: {avg_loss:.7f}")
            elif verbose > 1:
                print(f"Epoch {epoch+1}/{epochs} {(epoch+1)/epochs * 100:.1f}%\tLoss: {avg_loss:.7f}")

        loss = self.loss(y, y_pred)
        avg_loss = loss.mean()
        end = timer()

        if verbose >= 1:
            print("\n------------------------------------------------\n")
            print(f"Finished training if {end - start:.5f} seconds")
            print(f"Loss: {avg_loss:.7f}\n")

        return avg_loss
