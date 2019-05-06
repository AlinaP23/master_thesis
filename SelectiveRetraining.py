import numpy as np
import pandas as pd
import data_lib
from sklearn.neural_network import MLPClassifier
from sklearn.utils import check_array
from sklearn import model_selection
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.extmath import safe_sparse_dot
from copy import copy, deepcopy


class SelectiveRetrainingNetwork(MLPClassifier):
    def __init__(self, hidden_layer_sizes, learning_rate_init, random_state):
        self.retrain_pass = False
        self.position_missing = None
        self.weight_threshold = 0
        super().__init__(hidden_layer_sizes=hidden_layer_sizes,
                         learning_rate_init=learning_rate_init,
                         random_state=random_state)

    def selective_fit(self, features, labels, position_missing, weight_threshold):
        self.retrain_pass = True
        self.position_missing = position_missing
        self.weight_threshold = weight_threshold
        super().fit(features, labels)
        self.retrain_pass = False

    def _backprop(self, x, y, activations, deltas, coef_grads, intercept_grads):
        loss, coef_grads, intercept_grads = super()._backprop(x, y, activations, deltas, coef_grads, intercept_grads)
        if self.retrain_pass:
            # determine nodes & weights affected by missing feature
            affected_nodes = []
            affected_coefs = []
            affected_nodes.append([0] * len(activations[0][0]))
            affected_nodes[0][self.position_missing] = 1
            for layer in range(1, len(activations)):
                affected_nodes.append([0] * len(activations[layer][0]))
                layer_weights = []
                for node in range(0, len(self.coefs_[layer-1])):
                    if affected_nodes[layer - 1][node] == 1:
                        weights = [0] * len(self.coefs_[layer-1][node])
                        for x in range(0, len(self.coefs_[layer-1][node])):
                            if abs(self.coefs_[layer-1][node][x]) > self.weight_threshold:
                                weights[x] = 1
                                affected_nodes[layer][x] = 1
                    else:
                        weights = [0] * len(self.coefs_[layer-1][node])
                    layer_weights.append(weights)
                affected_coefs.append(layer_weights)

            # restrict coef & intercept adjustment to nodes affected to feature by setting all other grads to 0
            coef_grads = [x * y for x, y in zip(coef_grads, affected_coefs)]
            intercept_grads = [x * y for x, y in zip(intercept_grads, affected_nodes[1:])]

        return loss, coef_grads, intercept_grads


class SelectiveRetrainingCommittee:
    def __init__(self, learning_rate_init, hidden_layer_sizes, random_state):
        self.retrained_networks = None
        self.selective_network = SelectiveRetrainingNetwork(hidden_layer_sizes=hidden_layer_sizes,
                                                            learning_rate_init=learning_rate_init,
                                                            random_state=random_state)

    def fit(self, features, labels, weight_threshold):
        # train 'basic' neural network using the complete data set
        self.selective_network.fit(features, labels)
        self.retrained_networks = [self.selective_network] * len(features)

        # retrain one 'new' network for each missing feature (combination):
        # retrain 'original' network on an incomplete data set, selectively adjusting only nodes affected by the
        # missing feature(s)
        for i in range(0, len(features)):
            features_incomplete = np.copy(features)
            features_incomplete[:, i] = 0
            self.retrained_networks[i].selective_fit(features_incomplete, labels, i, weight_threshold)

    #def predict(self, points):
        # TODO


if __name__ == "__main__":
    data_set = "bank"
    test_size = 0.1
    random_state = 7
    X, Y, activation, labels, label_df = data_lib.get_dataset(data_set)
    x_train, x_test, y_train, y_test = \
        model_selection.train_test_split(X, Y, test_size=test_size, random_state=random_state)

    sr_learning_rate_init = 0.1
    sr_hidden_layer_sizes = [10, 10, 10]
    sr_random_state = 7
    sr_weight_threshold = 0.1

    selective_committee = SelectiveRetrainingCommittee(learning_rate_init=sr_learning_rate_init,
                                                       hidden_layer_sizes=sr_hidden_layer_sizes,
                                                       random_state=sr_random_state)

    selective_committee.fit(x_train, y_train, sr_weight_threshold)

