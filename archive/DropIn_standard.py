"""
Source: https://www.python-course.eu/neural_networks_with_python_numpy.php
"""

import numpy as np
from scipy.stats import truncnorm


@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)


activation_function = sigmoid


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


class NeuralNetwork:
    def __init__(self,
                 no_of_in_nodes,
                 no_of_out_nodes,
                 no_of_hidden_nodes,
                 learning_rate,
                 p_dropin):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate
        self.p_dropin = p_dropin
        self.create_weight_matrices()

    def create_weight_matrices(self):
        rad = 1 / np.sqrt(self.no_of_in_nodes)
        x = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_in_hidden = x.rvs((self.no_of_hidden_nodes, self.no_of_in_nodes))
        rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        x = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_out = x.rvs((self.no_of_out_nodes, self.no_of_hidden_nodes))

    def train(self, input_vector, target_vector):
        # input_vector and target_vector can be tuple, list or ndarray
        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T

        # DropIn implementation
        if type(self.p_dropin) is list:
            dropout_array = np.random.binomial(1, self.p_dropin)
        else:
            dropout_array = np.random.binomial(1, self.p_dropin, size=input_vector.shape)
        input_vector_dropout = input_vector * dropout_array

        output_vector1 = np.dot(self.weights_in_hidden, input_vector_dropout)
        output_vector_hidden = activation_function(output_vector1)

        output_vector2 = np.dot(self.weights_hidden_out, output_vector_hidden)
        output_vector_network = activation_function(output_vector2)

        output_errors = target_vector - output_vector_network
        # update the weights:
        tmp = output_errors * output_vector_network * (1.0 - output_vector_network)
        tmp = self.learning_rate * np.dot(tmp, output_vector_hidden.T)
        self.weights_hidden_out += tmp
        # calculate hidden errors:
        hidden_errors = np.dot(self.weights_hidden_out.T, output_errors)
        # update the weights:
        tmp = hidden_errors * output_vector_hidden * (1.0 - output_vector_hidden)
        self.weights_in_hidden += self.learning_rate * np.dot(tmp, input_vector_dropout.T)
        print("Nach Backprop:", self.weights_in_hidden)

    def run(self, input_vector):
        """
        running the network with an input vector input_vector.
        input_vector can be tuple, list or ndarray
        """
        # turning the input vector into a column vector
        input_vector = np.array(input_vector, ndmin=2).T
        output_vector = np.dot(self.weights_in_hidden, input_vector)
        output_vector = activation_function(output_vector)

        output_vector = np.dot(self.weights_hidden_out, output_vector)

        return output_vector


if __name__ == "__main__":
    data1 = [((3, 4, 5, 6, 7), (0.99, 0.01)), ((4.2, 5.3, 3, 3, 3), (0.99, 0.01)),
             ((4, 3, 5, 6, 7), (0.99, 0.01)), ((6, 5, 3, 3, 3), (0.99, 0.01)),
             ((4, 6, 3, 3, 3), (0.99, 0.01)), ((3.7, 5.8, 3, 3, 3), (0.99, 0.01)),
             ((3.2, 4.6, 3, 3, 3), (0.99, 0.01)), ((5.2, 5.9, 3, 3, 3), (0.99, 0.01)),
             ((5, 4, 3, 3, 3), (0.99, 0.01)), ((7, 4, 3, 3, 3), (0.99, 0.01)),
             ((3, 7, 3, 3, 3), (0.99, 0.01)), ((4.3, 4.3, 3, 3, 3), (0.99, 0.01))]
    data2 = [((-3, -4, 3, 3, 3), (0.01, 0.99)), ((-2, -3.5, 3, 3, 3), (0.01, 0.99)),
             ((-1, -6, 3, 3, 3), (0.01, 0.99)), ((-3, -4.3, 3, 3, 3), (0.01, 0.99)),
             ((-4, -5.6, 3, 3, 3), (0.01, 0.99)), ((-3.2, -4.8, 3, 3, 3), (0.01, 0.99)),
             ((-2.3, -4.3, 3, 3, 3), (0.01, 0.99)), ((-2.7, -2.6, 3, 3, 3), (0.01, 0.99)),
             ((-1.5, -3.6, 3, 3, 3), (0.01, 0.99)), ((-3.6, -5.6, 3, 3, 3), (0.01, 0.99)),
             ((-4.5, -4.6, 3, 3, 3), (0.01, 0.99)), ((-3.7, -5.8, 3, 3, 3), (0.01, 0.99))]
    data = data1 + data2
    np.random.shuffle(data)

    simple_network = NeuralNetwork(no_of_in_nodes=5,
                                   no_of_out_nodes=2,
                                   no_of_hidden_nodes=3,
                                   learning_rate=0.1,
                                   p_dropin=0.7)
    print(simple_network.weights_in_hidden)
    print(simple_network.weights_hidden_out)

    size_of_learn_sample = int(len(data)*0.9)
    learn_data = data[:size_of_learn_sample]
    test_data = data[-size_of_learn_sample:]
    print()

    number_of_train_iterations = 12
    for y in range(number_of_train_iterations):
        np.random.shuffle(learn_data)
        for i in range(size_of_learn_sample):
            point, label = learn_data[i][0], learn_data[i][1]
            simple_network.train(point, label)

    for i in range(size_of_learn_sample):
        point, label = learn_data[i][0], learn_data[i][1]
        cls1, cls2 = simple_network.run(point)
        print(point, cls1, cls2, end=": ")
        if cls1 > cls2:
            if label == (0.99, 0.01):
                print("class1 correct", label)
            else:
                print("class2 incorrect", label)
        else:
            if label == (0.01, 0.99):
                print("class 1 correct", label)
            else:
                print("class2 incorrect", label)
