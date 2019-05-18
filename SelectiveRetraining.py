import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from copy import deepcopy


class SelectiveRetrainingNetwork(MLPClassifier):
    def __init__(self, hidden_layer_sizes, learning_rate_init, random_state, activation):
        self.retrain_pass = False
        self.position_missing = None
        self.weight_threshold = 0
        super().__init__(hidden_layer_sizes=hidden_layer_sizes,
                         learning_rate_init=learning_rate_init,
                         random_state=random_state,
                         activation=activation)

    def selective_fit(self, features, labels, position_missing, weight_threshold):
        """ Triggering the selective retraining process. The network will be retrained on an incomplete training set
        (missing feature determined by position_missing). Only nodes, weights and intercepts which are 'affected' by
        the missing feature are readjusted.

        Parameters
        ----------
        features: array of shape [n_samples, n_features]
            Samples to be used for retraining of the NN
        labels: array of shape [n_samples]
            Labels for class membership of each sample
        position_missing: integer
            The position of the feature within the samples that shall be treated as 'missing' during the retraining
        weight_threshold: float
            Determines which of the NN's weights and nodes are considered as affected by the missing feature and
            readjusted after the backpropagation and loss calculation
        """
        self.retrain_pass = True
        self.position_missing = position_missing
        self.weight_threshold = weight_threshold
        super().fit(features, labels)
        self.retrain_pass = False

    def _backprop(self, x, y, activations, deltas, coef_grads, intercept_grads):
        """Compute the loss function and its corresponding derivatives with respect to each parameter: weights and bias
        vectors. In case this function is called during a selective retraining step, calculations are restricted to
        the derivatives for affected nodes only.

        Parameters
        ----------
        x : {array-like, sparse matrix}, shape [n_samples, n_features]
            The input data.
        y : array of shape [n_samples]
            The target values.
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
            The ith element contains the amount of change used to update the intercept parameters of the ith layer in
            an iteration.

        Returns
        -------
        loss : float
        coef_grads : list, length = n_layers - 1
        intercept_grads : list, length = n_layers - 1
        """
        if self.retrain_pass:
            # normalize weight threshold
            maxs = []
            mins = []
            for w1 in self.coefs_:
                maxs_temp = []
                mins_temp = []
                for w2 in w1:
                    maxs_temp.append(max(w2, key=abs))
                    mins_temp.append(min(w2, key=abs))
                maxs.append(max(maxs_temp, key=abs))
                mins.append(min(mins_temp, key=abs))

            weight_range = abs(max(maxs, key=abs)) - abs(min(mins, key=abs))
            normalized_threshold = abs(min(mins, key=abs)) + weight_range * self.weight_threshold

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
                        for m in range(0, len(self.coefs_[layer-1][node])):
                            if abs(self.coefs_[layer-1][node][m]) > normalized_threshold:
                                weights[m] = 1
                                affected_nodes[layer][m] = 1
                    else:
                        weights = [0] * len(self.coefs_[layer-1][node])
                    layer_weights.append(weights)
                affected_coefs.append(layer_weights)

            # restrict coef & intercept adjustment to nodes affected to by setting the deltas for all others to 0
            deltas = [d * n for d, n in zip(deltas, affected_nodes)]

        loss, coef_grads, intercept_grads = super()._backprop(x, y, activations, deltas, coef_grads, intercept_grads)

        return loss, coef_grads, intercept_grads


class SelectiveRetrainingCommittee:
    def __init__(self, learning_rate_init, hidden_layer_sizes, random_state, activation):
        self.retrained_networks = None
        self.selective_network = SelectiveRetrainingNetwork(hidden_layer_sizes=hidden_layer_sizes,
                                                            learning_rate_init=learning_rate_init,
                                                            random_state=random_state,
                                                            activation=activation)

    def fit(self, features, labels, weight_threshold):
        """ Triggering the committee training process. In a first step, an 'original' neural network is trained on the
        complete training data. Subsequently, for each of the features, the 'original' NN is copied and selectively
        retrained on the adjusted training data missing the respective feature.

        Parameters
        ----------
        features: array of shape [n_samples, n_features]
            Samples to be used for training of the NNs
        labels: array of shape [n_samples]
            Labels for class membership of each sample
        weight_threshold: float
            Determines which of the NN's weights and nodes are considered as affected by the missing feature and
            readjusted after the backpropagation and loss calculation (to be forwarded to the retraining process)
        """
        # train 'basic' neural network using the complete data set
        self.selective_network = self.selective_network.fit(features, labels)
        self.retrained_networks = []

        # retrain one 'new' network for each missing feature (combination):
        # retrain 'original' network on an incomplete data set, selectively adjusting only nodes affected by the
        # missing feature(s)
        for i in range(0, len(features[0])):
            print("Retraining network ", i + 1, " of ", len(features[0]))
            features_incomplete = np.copy(features)
            features_incomplete[:, i] = 0
            retrained_network = deepcopy(self.selective_network)
            retrained_network.selective_fit(features_incomplete, labels, i, weight_threshold)
            self.retrained_networks.append(retrained_network)

    def predict(self, points, data_frame=False, inverted_weights=None):
        """Classify the given input using the retraining committee. If no features is missing for a data point, use
        original network for classification only. If data is missing, classify the point with specific retrained
        classifiers for each of the missing features and return the majority vote (weighted if weights are given).

        Parameters
        ----------
        points: array of shape [n_samples, n_features]
            Samples to be classified
        data_frame: boolean
            Indicates whether the label array to be returned should be transformed to a data frame
        inverted_weights: None or array of shape [n_features]
            inverted weight of each feature, i.e. how much weight an algorithm missing the incorporation of this feature
            should have

        Returns
        ----------
        y_predicted: array of shape [n_samples]
            Predicted labels for the given points
        """
        y_predicted = []
        for p in range(0, len(points)):
            # determine if/which feature is missing
            index = np.where(points[p] == 0)
            if not index[0].size:
                # if no features is missing, use original network for classification
                y_predicted.append(self.selective_network.predict([points[p]])[0])
            else:
                # classify point with specific retrained classifiers for each of the missing features
                summed_results = [0] * self.selective_network.n_outputs_
                if inverted_weights is not None:
                    # calculate sum of inverted LRP weights for normalization purposes
                    summed_weights = [inverted_weights[x] for x in index[0]]
                    summed_weights = sum(summed_weights)
                for f in range(0, index[0].size):
                    results = self.retrained_networks[index[0][f]].predict_proba([points[p]])
                    if inverted_weights is not None:
                        # weight predictions according to inverted LRP scores
                        results = [x * (inverted_weights[index[0][f]]/summed_weights) for x in results]
                    summed_results = [x + y for x, y in zip(summed_results, results[0])]
                # determine weighted majority vote result
                prediction = [0] * self.selective_network.n_outputs_
                prediction[summed_results.index(max(summed_results))] = 1
                y_predicted.append(prediction)
        if data_frame:
            y_predicted = pd.DataFrame(y_predicted)
        return y_predicted

    def predict_without_retraining(self, points):
        """Classify the given input using the 'original' network trained on the complete data set only.

        Parameters
        ----------
        points: array of shape [n_samples, n_features]
            Samples to be classified

        Returns
        ----------
        y_predicted: array of shape [n_samples]
            Predicted labels for the given points
        """
        predictions = self.selective_network.predict(points)

        return predictions
