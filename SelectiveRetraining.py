import numpy as np
import pandas as pd
import data_lib
from sklearn.neural_network import MLPClassifier
from sklearn.utils import check_array
from sklearn import model_selection
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.extmath import safe_sparse_dot


class SelectiveRetrainingNetwork(MLPClassifier):
    def __init__(self, hidden_layer_sizes, learning_rate_init, random_state)
        self.retrain_pass = False
        super().__init__(hidden_layer_sizes=self.hidden_layer_sizes,
                         learning_rate_init=self.learning_rate_init,
                         random_state=self.random_state)

    def selective_fit(self, features, labels):
        self.retrain_pass = True
        super().fit(features, labels)
        self.retrain_pass = False

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


class SelectiveRetrainingCommittee:
    def __init__(self, learning_rate_init, hidden_layer_sizes, random_state):
        self.retrained_networks = None
        self.selective_network = SelectiveRetrainingNetwork(hidden_layer_sizes=hidden_layer_sizes,
                                                            learning_rate_init=learning_rate_init,
                                                            random_state=random_state)

    def fit(self, features, labels):
        # train 'basic' neural network using the complete data set
        self.selective_network.fit(features, labels)
        self.retrained_networks = [self.selective_network] * len(features)

        # retrain network on incomplete data set, selectively adjusting only nodes affected by the missing feature(s)
        for i in range(0, len(features)):
            features_incomplete = features
            features_incomplete[:, i] = 0
            self.retrained_networks[i].selective_fit(features_incomplete, labels)


