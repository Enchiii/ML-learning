import cupy as cp


# Dense layer
class Dense:
    def __init__(self, input_size: int, output_size: int):
        # init weights and biases
        self.weights: cp.ndarray = 0.01 * cp.random.randn(input_size, output_size)
        self.biases: cp.ndarray = cp.zeros((1, output_size))
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
        self.inputs: cp.ndarray | None = None
        self.output: cp.ndarray | None = None

    def __call__(self, inputs: cp.ndarray) -> cp.ndarray:
        return self.forward(inputs)

    # forward propagation
    def forward(self, inputs: cp.ndarray) -> cp.ndarray:
        exp_values = cp.exp(inputs - cp.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / cp.sum(exp_values, axis=1, keepdims=True)
        return self.output


# WARNING: not testes, may not work
# Sigmoid activation
class Sigmoid:
    def __init__(self):
        self.inputs: cp.ndarray | None = None
        self.output: cp.ndarray | None = None

    def __call__(self, inputs: cp.ndarray) -> cp.ndarray:
        return self.forward(inputs)

    def forward(self, inputs: cp.ndarray) -> cp.ndarray:
        self.inputs = inputs
        self.output = 1 / (1 + cp.exp(-inputs))
        return self.output
