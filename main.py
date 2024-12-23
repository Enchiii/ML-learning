# %%
import numpy as np
from sklearn.datasets import make_moons
from MlLib.layers import Dense, ReLU, Sigmoid, Softmax
from MlLib.losses import BinaryCrossEntropy, CategoricalCrossEntropy
import cupy as cp
from MlLib.models import Model
from MlLib.metrics import Metrics


if __name__ == "__main__":

    # %%
    X, y = make_moons(n_samples=10000, noise=0.2, random_state=42)

    X = cp.asarray(X)
    y = cp.asarray(y)

    y_one_hot = cp.column_stack((y, 1 - y))

    # %%
    model = Model(
        [Dense(2, 64, activation=ReLU()), Dense(64, 2, activation=Sigmoid())],
        loss=CategoricalCrossEntropy(),
        learning_rate=0.1,
    )

    loss = model.fit(X, y_one_hot, epochs=10, verbose=2)

    # %%
    pred_probs = model.predict(X)
    preds = cp.argmax(pred_probs, axis=1)

    metrics = Metrics()
    metrics(y.get(), preds.get())
    print(metrics.accuracy)
