"""
Source: https://www.python-course.eu/neural_networks_with_python_numpy.php
DropIn: https://arxiv.org/pdf/1705.02643.pdf
"""

import numpy as np
from sklearn.neural_network import MLPRegressor


class DropInNetworkRegression(MLPRegressor):
    def __init__(self,
                 learning_rate_init,
                 p_dropin,
                 hidden_layer_sizes,
                 random_state,
                 activation):
        self.seed = None
        self.train_pass = False
        self.dropout_arrays = []
        self.p_dropin = p_dropin
        super().__init__(hidden_layer_sizes=hidden_layer_sizes,
                         learning_rate_init=learning_rate_init,
                         random_state=random_state,
                         activation=activation)

    def _forward_pass(self, activations):
        """Perform a forward pass on the network by computing the values of the neurons in the hidden layers and the
        output layer. If performed during the training process, DropIn is implemented, i.e. single input nodes are
        masked to be ignored (during forward pass and in the following backpropagation step).

        Parameters
        ----------
        activations : list, length = n_layers - 1
            The ith element of the list holds the values of the ith layer.
        """
        # Test if dropin needs to be implemented as method is called during training
        if self.train_pass:
            self.dropout_arrays = [None] * len(activations[0])
            np.random.seed(self.seed)
            for j in range(len(activations[0])):
                # DropIn implementation
                if type(self.p_dropin) is list:
                    self.dropout_arrays[j] = np.random.binomial(1, self.p_dropin)
                else:
                    self.dropout_arrays[j] = np.random.binomial(1, self.p_dropin, size=activations[0][j].shape)
                activations[0][j] = activations[0][j] * self.dropout_arrays[j]
        super()._forward_pass(activations)

        return activations

    def fit_dropin(self, features_fit, labels_fit, np_seed):
        """ Triggers the training of the DropInNetwork.

        Parameters
        ----------
        features_fit: array of shape [n_samples, n_features]
            Samples to be used for training of the NN
        labels_fit:  array of shape [n_samples]
            Labels for class membership of each sample
        np_seed: integer
             Seed to make numpy randomization reproducible.
        """
        self.train_pass = True
        self.seed = np_seed
        super().fit(features_fit, labels_fit)
        self.train_pass = False

