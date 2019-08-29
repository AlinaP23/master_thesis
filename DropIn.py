"""
Source: https://www.python-course.eu/neural_networks_with_python_numpy.php
DropIn: https://arxiv.org/pdf/1705.02643.pdf
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils import shuffle


class DropInNetwork(MLPClassifier):
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
        self.sequence_length = 1
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
            dropout_array = [None]
            np.random.seed(self.seed)
            for j in range(len(activations[0])):
                # DropIn implementation
                if (j == 0) or (j % self.sequence_length == 0):
                    if type(self.p_dropin) is list:
                        dropout_array = np.random.binomial(1, self.p_dropin)
                    else:
                        dropout_array= np.random.binomial(1, self.p_dropin, size=activations[0][j].shape)
                self.dropout_arrays[j] = dropout_array
                activations[0][j] = activations[0][j] * self.dropout_arrays[j]
        super()._forward_pass(activations)

        return activations

    def _compute_loss_grad(self, layer, n_samples, activations, deltas, coef_grads, intercept_grads):
        """Compute the gradient of loss with respect to coefs and intercept for
        specified layer. This function does backpropagation for the specified one layer.

        Parameters
        ----------
        layer : integer
            Number of the layer for which the loss shall be calculated.
        n_samples : integer
            Number of samples for which the loss shall be calculated
        activations : list, length = n_layers - 1
            The ith element of the list holds the values of the ith layer.
        deltas : list, length = n_layers - 1
            The ith element of the list holds the difference between the activations of the i + 1 layer and the
            backpropagated error. More specifically, deltas are gradients of loss with respect to z in each layer, where
            z = wx + b is the value of a particular layer before passing through the activation function
        coef_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the coefficient parameters of the ith layer in
            an iteration.
        intercept_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the intercept parameters of the ith layer in an
            iteration.

        Returns
        ----------
        coef_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the coefficient parameters of the ith layer in
            an iteration.
        intercept_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the intercept parameters of the ith layer in an
            iteration.
        """

        coef_grads[layer] = safe_sparse_dot(activations[layer].T,
                                            deltas[layer])
        coef_grads[layer] += (self.alpha * self.coefs_[layer])
        coef_grads[layer] /= n_samples

        intercept_grads[layer] = np.mean(deltas[layer], 0)

        return coef_grads, intercept_grads

    def fit_dropin(self, features_fit, labels_fit, np_seed, epochs, sequence_length=None):
        """ Triggers the training of the DropInNetwork.

        Parameters
        ----------
        features_fit: array of shape [n_samples, n_features]
            Samples to be used for training of the NN
        labels_fit:  array of shape [n_samples]
            Labels for class membership of each sample
        np_seed: integer
             Seed to make numpy randomization reproducible.
        epochs: integer
            Number of maximum epochs, i.e. number of times the algorithm will be trained with the data set.
        sequence_length: integer
            In case of sequential data: number of instances grouped into a sequence
        """
        self.train_pass = True
        self.seed = np_seed
        np.random.seed(self.seed)
        no_instances = labels_fit.__len__()

        features_epochs = np.copy(features_fit)
        labels_epochs = np.copy(labels_fit)

        if sequence_length:
            self.sequence_length = sequence_length
            features_sequences = np.array([features_fit[0:self.sequence_length]])
            labels_sequences = np.array([labels_fit[0:self.sequence_length]])
            for i in range(1, (int(no_instances / self.sequence_length))):
                features_sequences = np.concatenate(
                    (features_sequences, np.array([features_fit[i * self.sequence_length:(i + 1) * self.sequence_length]])),
                    axis=0)
                labels_sequences = np.concatenate(
                    (labels_sequences, np.array([labels_fit[i * self.sequence_length:(i + 1) * self.sequence_length]])), axis=0)
            c = list(zip(features_sequences, labels_sequences))
        else:
            c = list(zip(features_fit, labels_fit))

        for i in range(1, epochs):
            # create shuffled multiple epoch data set
            shuffled_c = shuffle(c)
            features, labels = zip(*shuffled_c)
            if sequence_length:
                features = np.array(features)
                features = features.reshape(-1, features.shape[-1])
                labels = np.ravel(np.array(labels))
            features_epochs = np.concatenate((features_epochs, features))
            labels_epochs = np.concatenate((labels_epochs, labels))
        super().fit(features_epochs, labels_epochs)

        # reset parameter
        self.train_pass = False
