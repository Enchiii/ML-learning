from sklearn.datasets import make_moons
from MlLib.layers import Dense, ReLU, Sigmoid
from MlLib.losses import CategoricalCrossEntropy
import cupy as cp
from MlLib.RL.model import RLModel

if __name__ == "__main__":
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

    X = cp.asarray(X)
    y = cp.asarray(y)

    model = RLModel(
        [
            Dense(2, 16, activation=ReLU()),
            Dense(16, 1, activation=Sigmoid()),
        ],
        loss=CategoricalCrossEntropy(),
        learning_rate=0.01,
    )

    loss = model.fit(X, y, epochs=10, verbose=2)
