"""
Source: https://www.python-course.eu/neural_networks_with_python_numpy.php
DropIn: https://arxiv.org/pdf/1705.02643.pdf
"""

from scipy.stats import truncnorm
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.neural_network._base import LOSS_FUNCTIONS, DERIVATIVES
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
        self.dropout_arrays = []
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
            self.dropout_arrays = [None] * len(activations[0])
            for i in range(len(activations[0]) - 1):
                # DropIn implementation
                if type(self.p_dropin) is list:
                    self.dropout_arrays[i] = np.random.binomial(1, self.p_dropin)
                else:
                    self.dropout_array[i] = np.random.binomial(1, self.p_dropin, size=activations[0][i].shape)
                activations[0][i] = activations[0][i] * self.dropout_arrays[i]

        super()._forward_pass(activations)

        return activations

    def _compute_loss_grad(self, layer, n_samples, activations, deltas,
                           coef_grads, intercept_grads):
        """Compute the gradient of loss with respect to coefs and intercept for
        specified layer.
        This function does backpropagation for the specified one layer.
        """

        coef_grads[layer] = safe_sparse_dot(activations[layer].T,
                                            deltas[layer])
        coef_grads[layer] += (self.alpha * self.coefs_[layer])
        coef_grads[layer] /= n_samples

        intercept_grads[layer] = np.mean(deltas[layer], 0)

        return coef_grads, intercept_grads

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
                                   p_dropin=[0.9, 0.1, 0.9, 0.1])

    dropin_network.fit_dropin(x_train, y_train)
