import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.utils import check_array
from sklearn import model_selection
from sklearn.metrics import accuracy_score


class CustomMLPClassifier(MLPClassifier):

    def predict_lrp(self, data):
        """ https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/neural_network/multilayer_perceptron.py
        Predict and calculate LRP values using the trained model.
        Altered scikit-method (sole difference: output activations additionally)

        Parameters
        ----------
        data : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.

        Returns
        ----------
        y_prediction : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The decision function of the samples for each class in the model.
        activations :  array of shape [m=no_layers, n=no_nodes]
            Node activations
        """
        data = check_array(data, accept_sparse=['csr', 'csc', 'coo'])

        # Make sure self.hidden_layer_sizes is a list
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        layer_units = [data.shape[1]] + hidden_layer_sizes + [self.n_outputs_]

        # Initialize layers
        activations = [data]

        for i in range(self.n_layers_ - 1):
            activations.append(np.empty((data.shape[0], layer_units[i + 1])))
        # forward propagate
        self._forward_pass(activations)
        y_predicted = activations[-1]

        return y_predicted, activations


class LRPNetwork:
    def __init__(self,
                 hidden_layer_sizes,
                 learning_rate_init,
                 no_of_in_nodes,
                 activation):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init
        self.no_of_in_nodes = no_of_in_nodes
        self.activation = activation
        self.LRP_scores_regarded = 0

    def avg_lrp_score_per_feature(self, features, labels, test_size, seed, random_states, alpha, accuracy_threshold):
        """ Calculates the average LRP score per feature based on the calculations of several individual networks.

        Parameters
        ----------
        features: array of shape [n_samples, n_features]
            Samples to be used for the calculation of the LRP scores
        labels: array of shape [n_samples]
            Labels for class membership of each sample
        test_size: float
            Percentage of the samples to be used for testing purposes
        seed: integer
            Seed to make numpy randomization reproducible.
        random_states: list
            Determine random number generation. Pass an int for reproducible output across multiple function calls.
        alpha: integer
            Determines the weighting of positive and negative influences on a node during LRP score calculation.
            (alpha = weighting of positive values; beta = weighting of negative values (alpha - 1))
        accuracy_threshold: float
            Determines accuracy threshold a neural network has to achieve to be used during LRP score calculation

        Returns
        ----------
        avg_feature_lrp_scores: array of shape [n_features]
            Lists the average LRP scores per feature
        highest_performing_network: instance of CustomMLPClassifier
            The network which scored the highest accuracy
        """
        avg_feature_lrp_scores = [0] * self.no_of_in_nodes
        iterations = len(random_states)
        single_networks = [None] * iterations
        accuracies = [0] * iterations
        network_results = 0

        for i in range(0, iterations):
            print("Iteration:", i + 1)
            single_network_results, single_networks[i], accuracies[i] = \
                self.single_network_avg_lrp_score_per_feature(features, labels, test_size, seed, random_states[i],
                                                              alpha, accuracy_threshold)
            if single_network_results is not None:
                avg_feature_lrp_scores = [x + y for x, y in zip(avg_feature_lrp_scores, single_network_results)]
                network_results += 1

        if network_results != 0:
            avg_feature_lrp_scores[:] = [x / network_results for x in avg_feature_lrp_scores]

        highest_performing_network = single_networks[accuracies.index(max(accuracies))]

        return avg_feature_lrp_scores, highest_performing_network

    def single_network_avg_lrp_score_per_feature(self, features, labels, test_size, seed, random_state, alpha,
                                                 threshold):
        """ Calculates the average LRP score per feature within one network.

        Parameters
        ----------
        features: array of shape [n_samples, n_features]
            Samples to be used for the calculation of the LRP scores
        labels: array of shape [n_samples]
            Labels for class membership of each sample
        test_size: float
            Percentage of the samples to be used for testing purposes
        seed: integer
            Seed to make numpy randomization reproducible.
        random_state: list
            Determines random number generation. Pass an int for reproducible output across multiple function calls.
        alpha: integer
            Determines the weighting of positive and negative influences on a node during LRP score calculation.
            (alpha = weighting of positive values; beta = weighting of negative values (alpha - 1)). Must be >= 0.
        threshold: float
            Determines accuracy threshold a neural network has to achieve to be used during LRP score calculation

        Returns
        ----------
        avg_feature_lrp_scores: array of shape [n_features]
            Lists the average LRP scores per feature
        mlp_network: instance of CustomMLPClassifier
            The newly trained network
        accuracy: float
            The newly trained network's accuracy score
        """
        # variable definition
        avg_feature_lrp_scores = [0] * self.no_of_in_nodes

        x_train, x_test, y_train, y_test = \
            model_selection.train_test_split(features, labels, test_size=test_size, random_state=seed, stratify=labels)

        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.values
        elif isinstance(y_test, pd.Series):
            y_test = pd.Series.tolist(y_test)

        # train neural network
        mlp_network = CustomMLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes,
                                          learning_rate_init=self.learning_rate_init,
                                          activation=self.activation,
                                          random_state=random_state)
        mlp_network.fit(x_train, y_train)

        predictions = mlp_network.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)

        if accuracy > threshold:
            # calculate avg. LRP scores for features - use correctly classified test data to determine LRP scores
            lrp_iterations = 0

            for j in range(0, len(y_test)):
                print("LRP Calculation ", j, " of ", len(y_test))
                self.LRP_scores_regarded += 1
                if ((isinstance(y_test[j], list) or isinstance(y_test[j], np.ndarray)) and y_test[j].all() == predictions[j].all()) or y_test[j] == predictions[j]:
                    lrp_iterations += 1
                    feature_lrp_scores = self.lrp_scores(mlp_network, [x_test[j]], alpha, alpha - 1)
                    avg_feature_lrp_scores = [x + y for x, y in zip(avg_feature_lrp_scores, feature_lrp_scores)]

            if lrp_iterations != 0:
                avg_feature_lrp_scores[:] = [x / lrp_iterations for x in avg_feature_lrp_scores]

            return avg_feature_lrp_scores, mlp_network, accuracy
        else:
            return None, mlp_network, accuracy

    def lrp_scores(self, network, data, alpha, beta):
        """ Calculates the relevance matrix/LRP score for all nodes during the classification of one sample.

        Parameters
        ----------
        network: instance of CustomMLPClassifier
            The classifier to be used to calculate the LRP scores
        data: array of shape [n_features]
            The data sample which shall be used to calculate the LRP scores
        alpha: integer
            Determines the weighting of positive influences on a node during LRP score calculation. Must be >= 0.
        beta: integer
            Determines the weighting of negative influences on a node during LRP score calculation. Must be alpha - 1.

        Returns
        ----------
        relevance_matrix: array of shape [n_layers, n_nodes]
            Contains the LRP scores for the neural networks nodes, i.e. the relevance of each node during the
            classification of the given data
        """
        y_predicted, activation_matrix = network.predict_lrp(data)
        # prepare y_predicted so that only relevance of most probable class is distributed
        y_predicted[np.where(y_predicted != np.max(y_predicted))] = 0
        relevance_matrix = list()
        relevance_matrix.append(y_predicted[0])

        # loop over layers
        for layer_n in range(network.n_layers_ - 1, 0, -1):
            if layer_n == 1:
                no_of_nodes = self.no_of_in_nodes
            else:
                no_of_nodes = self.hidden_layer_sizes[layer_n - 2]
            weight_matrix = network.coefs_[layer_n - 1]
            layer_relevance = [0] * no_of_nodes

            # loop over nodes
            for node_n in range(0, no_of_nodes):
                positive_relevance = 0
                negative_relevance = 0

                # calculate relevance input per connected node in higher layer
                for connection_n in range(0, len(weight_matrix[node_n])):
                    if weight_matrix[node_n, connection_n] > 0:
                        # j = node in lower layer; k = node in higher layer
                        pos_effect_j_on_k = activation_matrix[layer_n - 1][0][node_n] \
                                            * weight_matrix[node_n, connection_n]
                        # calculate the excitatory effects on node k in upper layer
                        pos_sum_effects_on_k = 0
                        for i in range(0, no_of_nodes):
                            if weight_matrix[i, connection_n] > 0:
                                pos_sum_effects_on_k += activation_matrix[layer_n - 1][0][i] \
                                                        * weight_matrix[i, connection_n]
                        # calculate the positive relevance of node j
                        if pos_effect_j_on_k != 0:
                            positive_relevance += relevance_matrix[0][connection_n] \
                                                  * (pos_effect_j_on_k / pos_sum_effects_on_k)
                    elif weight_matrix[node_n, connection_n] < 0:
                        # j = node in lower layer; k = node in higher layer
                        neg_effect_j_on_k = activation_matrix[layer_n - 1][0][node_n] \
                                                 * weight_matrix[node_n, connection_n]
                        # calculate the inhibitory effects on node k in upper layer
                        neg_sum_effects_on_k = 0
                        for i in range(0, no_of_nodes):
                            if weight_matrix[i, connection_n] < 0:
                                neg_sum_effects_on_k += activation_matrix[layer_n - 1][0][i] \
                                                        * weight_matrix[i, connection_n]
                        # calculate the negative relevance of node j
                        if neg_effect_j_on_k != 0:
                            negative_relevance += relevance_matrix[0][connection_n] \
                                                  * (neg_effect_j_on_k / neg_sum_effects_on_k)

                # weighting between positive and negative relevance
                layer_relevance[node_n] = positive_relevance * alpha - negative_relevance * beta

            relevance_matrix.insert(0, layer_relevance)

        return relevance_matrix[0]

    @staticmethod
    def lrp_scores_to_percentage(average_lrp_scores):
        """ Converts the raw LRP scores to percentages. To be used as input to Learn++.
            The inverted values will be used as input to Selective Retraining.

        Parameters
        ----------
        average_lrp_scores: array of shape [n_features]
            Average LRP scores per feature to be normalized to percentages

        Returns
        ----------
        average_lrp_scores_normalized: array of shape [n_features]
            Average LRP scores per feature normalized to percentages
        average_lrp_scores_normalized_inverted: array of shape [n_features]
            Inverted average LRP scores per feature normalized to percentage
        """
        average_lrp_scores = [abs(x) for x in average_lrp_scores]
        sum_lrp_scores = sum(average_lrp_scores)
        average_lrp_scores_normalized = [round(x / sum_lrp_scores, 5) for x in average_lrp_scores]

        average_lrp_scores_normalized_inverted = [sum_lrp_scores - x for x in average_lrp_scores]
        sum_lrp_scores_inverted = sum(average_lrp_scores_normalized_inverted)
        average_lrp_scores_normalized_inverted = \
            [round(x / sum_lrp_scores_inverted, 5) for x in average_lrp_scores_normalized_inverted]

        return average_lrp_scores_normalized, average_lrp_scores_normalized_inverted

    @staticmethod
    def lrp_scores_to_scaled(average_lrp_scores, threshold_max):
        """ Sets a certain maximum probability threshold to the feature with the highest LRP score and scales all
        further features' scores accordingly. Must be inverted (i.e. 1 - scaled score) to be used as input to DropIn.

        Parameters
        ----------
        average_lrp_scores: array of shape [n_features]
            Average LRP scores per feature to be scaled
        threshold_max: float
            Maximum probability threshold to be assigned to the feature with the highest LRP score

        Returns
        ----------
        average_lrp_scores_scaled: array of shape [n_features]
            Scaled average LRP scores
        average_lrp_scores_scaled_inverted: array of shape [n_features]
            Inverted scaled average LRP scores (DropIn probabilities)
        """
        average_lrp_scores = [abs(x) for x in average_lrp_scores]
        max_score = max(average_lrp_scores)
        average_lrp_scores_scaled = [x / max_score * threshold_max for x in average_lrp_scores]
        average_lrp_scores_scaled_inverted = [1 - x for x in average_lrp_scores_scaled]
        return average_lrp_scores_scaled, average_lrp_scores_scaled_inverted

    @staticmethod
    def lrp_scores_to_scaled_range(average_lrp_scores, threshold_max, threshold_min):
        """ Sets a certain maximum probability threshold to the feature with the highest LRP score and a minimum prob.
        threshold to the feature with the lowest LRP score. All further features' scores are scaled accordingly.
        Must be inverted (i.e. 1 - scaled score) to be used as input to DropIn.

        Parameters
        ----------
        average_lrp_scores: array of shape [n_features]
            Average LRP scores per feature to be scaled within the given range (min_threshold to max_threshold)
        threshold_max: float
            Maximum probability threshold to be assigned to the feature with the highest LRP score
        threshold_min: float
            Minimum probability threshold to be assigned to the feature with the lowest LRP score

        Returns
        ----------
        average_lrp_scores_ranged: array of shape [n_features]
            Average LRP scores per feature scaled by range
        average_lrp_scores_scaled_inverted: array of shape [n_features]
            Inverted average LRP scores per feature scaled by range
        """
        average_lrp_scores = [abs(x) for x in average_lrp_scores]
        max_score = max(average_lrp_scores)
        min_score = min(average_lrp_scores)
        lrp_range = max_score - min_score
        threshold_range = threshold_max - threshold_min
        average_lrp_scores_range = \
            [(x - min_score) / lrp_range * threshold_range + threshold_min for x in average_lrp_scores]
        average_lrp_scores_range_inverted = [1 - x for x in average_lrp_scores_range]
        return average_lrp_scores_range, average_lrp_scores_range_inverted
