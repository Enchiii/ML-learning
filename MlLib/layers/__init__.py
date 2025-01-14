import cupy as cp


# Dense layer
class Dense:
    def __init__(self, input_size: int, output_size: int, trainable=True):
        # init weights and biases
        self.weights: cp.ndarray = 0.01 * cp.random.randn(input_size, output_size)
        self.biases: cp.ndarray = cp.zeros((1, output_size))
        self.trainable = trainable
        # init variables
        self.inputs: cp.ndarray | None = None
        self.output: cp.ndarray | None = None
        self.dweights: cp.ndarray | None = None
        self.dbiases: cp.ndarray | None = None
        self.dinputs: cp.ndarray | None = None

    def __call__(self, inputs: cp.ndarray) -> cp.ndarray:
        return self.forward(inputs)

    # forward propagation
    def forward(self, inputs: cp.ndarray) -> cp.ndarray:
        self.inputs = inputs
        self.output = cp.dot(inputs, self.weights) + self.biases
        return self.output

    # backward propagation (calculating gradients)
    def backward(self, dvalues: cp.ndarray) -> cp.ndarray:
        self.dweights = cp.dot(self.inputs.T, dvalues)
        self.dbiases = cp.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = cp.dot(dvalues, self.weights.T)
        return self.dinputs


# Relu activation
class ReLU:
    def __init__(self):
        self.trainable = False
        # init variables
        self.inputs: cp.ndarray | None = None
        self.output: cp.ndarray | None = None
        self.dinputs: cp.ndarray | None = None

    def __call__(self, inputs: cp.ndarray) -> cp.ndarray:
        return self.forward(inputs)

    # forward propagation
    def forward(self, inputs: cp.ndarray) -> cp.ndarray:
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = cp.maximum(0, inputs)
        return self.output

    # backward propagation
    def backward(self, dvalues: cp.ndarray) -> cp.ndarray:
        self.dinputs = dvalues.copy()
        # zero gradient where input values are negative
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs


# Softmax activation
class Softmax:
    def __init__(self):
        self.trainable = False
        self.inputs: cp.ndarray | None = None
        self.output: cp.ndarray | None = None

    def __call__(self, inputs: cp.ndarray) -> cp.ndarray:
        return self.forward(inputs)

    # forward propagation
    def forward(self, inputs: cp.ndarray) -> cp.ndarray:
        exp_values = cp.exp(inputs - cp.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / cp.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, y_true: cp.ndarray, dvalues: cp.ndarray) -> cp.ndarray:
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


# WARNING: not testes, may not work
# Sigmoid activation
class Sigmoid:
    def __init__(self):
        self.trainable = False
        self.inputs: cp.ndarray | None = None
        self.output: cp.ndarray | None = None

    def __call__(self, inputs: cp.ndarray) -> cp.ndarray:
        return self.forward(inputs)

    def forward(self, inputs: cp.ndarray) -> cp.ndarray:
        self.inputs = inputs
        self.output = 1 / (1 + cp.exp(-inputs))
        return self.output
