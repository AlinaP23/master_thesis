"""
Source: https://www.python-course.eu/neural_networks_with_python_numpy.php
DropIn: https://arxiv.org/pdf/1705.02643.pdf
"""

from scipy.stats import truncnorm
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection
from sklearn.metrics import classification_report, accuracy_score


@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)

@np.vectorize
def relu(x):
    if x > 0:
        return x
    else:
        return 0


activation_function = relu


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


class DropInNetwork(MLPClassifier):
    def __init__(self,
                 learning_rate_init,
                 p_dropin,
                 hidden_layer_sizes):
        self.train_pass = False
        self.dropout_array = []
        self.p_dropin = p_dropin
        super().__init__(hidden_layer_sizes=hidden_layer_sizes,
                         learning_rate_init=learning_rate_init)

    def _forward_pass(self, activations):
        """Perform a forward pass on the network by computing the values
        of the neurons in the hidden layers and the output layer.
        Parameters
        ----------
        activations : list, length = n_layers - 1
            The ith element of the list holds the values of the ith layer.
        """
        # Test if dropin needs to be implemented as method is called during training
        if self.train_pass:
            # DropIn implementation
            if type(self.p_dropin) is list:
                self.dropout_array = np.random.binomial(1, self.p_dropin)
            else:
                self.dropout_array = np.random.binomial(1, self.p_dropin, size=activations[0].shape)
            activations[0] = activations[0] * self.dropout_array

        super()._forward_pass(activations)

        return activations

    def _backprop(self, X, y, activations, deltas, coef_grads,
                  intercept_grads):
        """Compute the MLP loss function and its corresponding derivatives
        with respect to each parameter: weights and bias vectors.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.
        y : array-like, shape (n_samples,)
            The target values.
        activations : list, length = n_layers - 1
             The ith element of the list holds the values of the ith layer.
        deltas : list, length = n_layers - 1
            The ith element of the list holds the difference between the
            activations of the i + 1 layer and the backpropagated error.
            More specifically, deltas are gradients of loss with respect to z
            in each layer, where z = wx + b is the value of a particular layer
            before passing through the activation function
        coef_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            coefficient parameters of the ith layer in an iteration.
        intercept_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            intercept parameters of the ith layer in an iteration.
        Returns
        -------
        loss : float
        coef_grads : list, length = n_layers - 1
        intercept_grads : list, length = n_layers - 1
        """
        loss, coef_grads, intercept_grads = super()._backprop(X, y, activations, deltas, coef_grads,
                                                              intercept_grads)

        # Test if dropin needs to be implemented as method is called during training
        if self.train_pass:
            coef_grads[0] = coef_grads[0] * self.dropout_array
            intercept_grads[0] = intercept_grads[0] * self.dropout_array

        return loss, coef_grads, intercept_grads

    def fit_dropin(self, features, labels):
        self.train_pass = True
        super().fit(features, labels)
        self.train_pass = False
        """
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
        """

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
    iris = pd.read_csv('./data/iris.csv')

    # Create numeric classes for species (0,1,2)
    iris.loc[iris['species'] == 'virginica', 'species'] = 0
    iris.loc[iris['species'] == 'versicolor', 'species'] = 1
    iris.loc[iris['species'] == 'setosa', 'species'] = 2

    # Create Input and Output columns
    X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    Y = iris[['species']].values.ravel()

    x_train, x_test, y_train, y_test = \
        model_selection.train_test_split(X, Y, test_size=0.1, random_state=7)

    dropin_network = DropInNetwork(hidden_layer_sizes=[10, 10, 10],
                                   learning_rate_init=0.1,
                                   p_dropin=[0.9, 0.1, 0.9, 0.9, 0.1])

    dropin_network.fit_dropin(x_train, y_train)
