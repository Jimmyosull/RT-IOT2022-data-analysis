import numpy as np



def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - np.tanh(x)**2


def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size


class Layer:
    def __init__(self) -> None:
        self.input = None
        self.output = None

    def forward_prop(self, input):
        raise NotImplementedError

    def backward_prop(self, input, lr):
        raise NotImplementedError



class FullyConnected(Layer):
    def __init__(self, input_size, output_size) -> None:
        self.weights = np.random.rand(input_size, output_size) - .5
        self.bias = np.random.rand(1, output_size) - .5


    def forward_prop(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output


    def backward_prop(self, output_err, lr):
        input_err = np.dot(output_err, self.weights.T)
        weights_err = np.dot(self.input.T, output_err)
        self.weights -= lr * weights_err
        self.bias -= lr * output_err
        return input_err



class Activation(Layer):
    def __init__(self, activation, activation_prime) -> None:
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime


    def forward_prop(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output


    def backward_prop(self, output_err, lr):
        return self.activation_prime(self.input) * output_err



class Network:
    def __init__(self) -> None:
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.losses = []
    

    def add(self, layer):
        self.layers.append(layer)
    

    def use_loss(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    
    def predict(self, input_data):
        results = []

        for i, _ in enumerate(input_data):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_prop(output)
            results.append(output)

        return results


    def fit(self, x_train, y_train, epochs = 250, lr = .05):
        for i in range(epochs):
            err = 0
            for j, _ in enumerate(x_train):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_prop(output)
                
                err += self.loss(y_train[j], output)

                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_prop(error, lr)
            
            err /= len(x_train)
            self.losses.append(err)
            print(f"epoch {i}/{epochs}      error={err}")
