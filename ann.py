import numpy as np


class ANN:
    """
    An artificial neural network
    """

    def __init__(self, input_num, hidden_num, epoch, batches, learning_rate, momentum):
        self.input_num = input_num  # Number of input nodes
        self.hidden_num = hidden_num    # Number of hidden nodes
        self.epoch = epoch  # Number of epochs
        self.learning_rate = learning_rate  # Learning rate
        self.momentum = momentum    # Momentum
        self.batches = batches   # Number of batches to split training set into
        self.weights1 = np.random.uniform(low=-1.0, high=1.0, size=(input_num, hidden_num))   # Initialize weights for layer 1
        self.weights2 = np.random.uniform(low=-1.0, high=1.0, size=(hidden_num, 1))    # Initialize weights for layer 2

    def sigmoid_prime(self, activation):
        """
        Derivative of sigmoid function takes activation as input
        """
        return self.sigmoid_function(activation) * (1 - self.sigmoid_function(activation))

    @staticmethod
    def sigmoid_function(activation):
        """
        Sigmoid function takes activation as input
        """
        return 1 / (1 + np.exp(-activation))

    @staticmethod
    def calc_activation(inputs, weights):
        """
        Calculate activation with given inputs and weights
        """
        return np.dot(inputs, weights)

    def predict(self, x_data):
        """
        Returns class prediction from given inputs
        """
        # Feedforward
        layer0 = x_data
        layer1 = self.sigmoid_function(self.calc_activation(layer0, self.weights1))
        layer2 = self.sigmoid_function(self.calc_activation(layer1, self.weights2))
        return np.round(layer2)

    def train(self, x_data, y_data):
        """
        Trains the neural network on a dataset using backpropagation
        """
        layer0 = np.array(np.array_split(x_data, self.batches))
        desired = np.array(np.array_split(y_data, self.batches))
        sum_square_error = np.Infinity
        layer1_delta_prev = 0
        layer2_delta_prev = 0
        e = 1
        while e < self.epoch + 1 and sum_square_error > 0.1:
            for b in range(self.batches):
                # Feedforward
                layer1 = self.sigmoid_function(self.calc_activation(layer0[b], self.weights1))
                layer2 = self.sigmoid_function(self.calc_activation(layer1, self.weights2))

                # Backpropagation
                layer2_delta = (desired[b] - layer2) * self.sigmoid_prime(layer2) + self.momentum*layer2_delta_prev
                layer1_delta = ((desired[b] - layer2) * self.sigmoid_prime(layer2)).dot(self.weights2.T) * self.sigmoid_prime(layer1) + self.momentum * layer1_delta_prev
                layer2_delta_prev = layer2_delta
                layer1_delta_prev = layer1_delta
                self.weights2 += self.learning_rate * layer1.T.dot(layer2_delta)
                self.weights1 += self.learning_rate*layer0[b].T.dot(layer1_delta)

                # Loop update
                sum_square_error = np.sum(np.square(desired[b] - layer2))
                e += 1
