import cupy as cp

cp.random.seed(42)


class Dense:
    def __init__(self, input_size: list | tuple | int, output_size: list | tuple | int, activation):
        # self.weights: cp.ndarray = cp.random.randn(output_size, input_size)
        self.weights: cp.ndarray = cp.random.randn(input_size, output_size) * 0.001
        self.biases: cp.ndarray = cp.zeros((1, output_size))
        self.activation = activation
        self.inputs = None
        self.outputs = None

    def forward(self, inputs: cp.ndarray) -> cp.ndarray:
        self.inputs = inputs
        self.outputs = cp.dot(inputs, self.weights) + self.biases
        self.outputs = self.activation.forward(self.outputs)
        return self.outputs

    def backward(self, error: cp.ndarray, learning_rate: float) -> cp.ndarray:
        if error.ndim == 1:
            error = error.reshape(-1, self.weights.shape[1])  # Ensure `error` has correct dimensions

        self.weights -= cp.dot(self.inputs.T, error) * learning_rate
        self.biases -= cp.sum(error, axis=0, keepdims=True) * learning_rate
        return cp.dot(error, self.weights.T)  # calc and return error for next layer


class ReLU:
    def __init__(self):
        self.inputs = None
        self.outputs: float = 0.0

    def forward(self, inputs: cp.ndarray) -> cp.ndarray:
        self.inputs = inputs
        self.outputs = cp.maximum(0.0, inputs)
        return self.outputs

    def backward(self, error: cp.ndarray, learning_rate: float) -> cp.ndarray:
        error[self.inputs <= 0.0] = 0.0
        return error


class Softmax:
    def __init__(self):
        self.outputs = None
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        exp_values = cp.exp(inputs - cp.max(inputs, axis=-1, keepdims=True))
        sum_of_exp_values = cp.sum(exp_values, axis=-1, keepdims=True)
        self.outputs = exp_values / sum_of_exp_values
        return self.outputs

    def backward(self, error: cp.ndarray, learning_rate: float) -> cp.ndarray:
        num_samples, num_classes = error.shape
        inputs_derivative = cp.zeros_like(error)

        for i in range(num_samples):
            single_output = self.outputs[i].reshape(-1, 1)
            jacobian_matrix = cp.diagflat(single_output) - cp.dot(single_output, single_output.T)
            # Compute gradient for this sample
            inputs_derivative[i] = cp.dot(jacobian_matrix, error[i])

        return inputs_derivative


class Sigmoid:
    def __init__(self):
        self.inputs = None
        self.outputs: float = 0.0

    def forward(self, inputs: cp.ndarray) -> cp.ndarray:
        self.inputs = inputs
        self.outputs = 1 / (1 + cp.exp(-inputs))  # Sigmoid activation
        return self.outputs

    def backward(self, error: cp.ndarray, learning_rate: float) -> cp.ndarray:
        return error * self.outputs * (1 - self.outputs)
