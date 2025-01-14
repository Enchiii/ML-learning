import cupy as cp


class Optimizer:
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        if not self.learning_rate or self.learning_rate < 0:
            raise ValueError("Learning rate must be greater than 0.")

    def update_params(self, layers):
        pass


# SGD optimizer
class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.01):
        super(SGD, self).__init__(learning_rate)
        self.learning_rate = learning_rate
        if not self.learning_rate or self.learning_rate < 0:
            raise ValueError("Learning rate must be greater than 0.")

    def update_params(self, layers):
        for layer in layers:
            if layer.trainable:
                layer.weights -= self.learning_rate * layer.dweights
                layer.biases -= self.learning_rate * layer.dbiases
