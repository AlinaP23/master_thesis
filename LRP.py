import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.utils import check_array
from sklearn import model_selection
from sklearn.metrics import classification_report


class CustomMLPClassifier(MLPClassifier):

    def predict_lrp(self, data):
        """Predict and calculate LRP values using the trained model
           Altered scikit-method (sole difference: output activations additionally)

        Parameters
        ----------
        data : {array-like, sparse matrix}, shape (n_samples, n_features)
                The input data.
        Returns
        -------
        y_prediction : array-like, shape (n_samples,) or (n_samples, n_outputs)
                       The decision function of the samples for each class in the model.
        activations : node activations (m=no_layers, n=no_nodes)
        """
        data = check_array(data, accept_sparse=['csr', 'csc', 'coo'])

        # Make sure self.hidden_layer_sizes is a list
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        layer_units = [data.shape[1]] + hidden_layer_sizes + \
            [self.n_outputs_]

        # Initialize layers
        activations = [data]

        for i in range(self.n_layers_ - 1):
            activations.append(np.empty((data.shape[0],
                                         layer_units[i + 1])))
        # forward propagate
        self._forward_pass(activations)
        y_predicted = activations[-1]

        return y_predicted, activations


class LRPNetwork:
    def __init__(self,
                 hidden_layer_sizes,
                 learning_rate_init,
                 no_of_in_nodes):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init
        self.no_of_in_nodes = no_of_in_nodes

    def avg_lrp_score_per_feature(self, features, labels, test_size, seed):
        # variable definition
        avg_feature_lrp_scores = [0] * self.no_of_in_nodes

        x_train, x_test, y_train, y_test = \
            model_selection.train_test_split(features, labels, test_size=test_size, random_state=seed)
        # train neural network
        mlp_network = CustomMLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes,
                                          learning_rate_init=self.learning_rate_init)
        mlp_network.fit(x_train, y_train)

        """
        # save the model to disk
        filename = 'lrp_network.nn'
        pickle.dump(mlp_network, open(filename, 'wb'))
        lrp_network = model_io.read(filename)
        """
        predictions = mlp_network.predict(x_test)
        print(classification_report(y_test, predictions))

        # calculate avg. LRP scores for features - use correctly classified test data to determine LRP scores
        lrp_iterations = 0

        for j in range(0, len(y_test)):
            if y_test[j] == predictions[j]:
                lrp_iterations += 1

                """
                # prepare initial relevance to reflect the model's dominant prediction
                # (ie depopulate non-dominant output neurons)
                mask = np.zeros_like(y_predicted)
                mask[:, np.argmax(y_predicted)] = 1
                r_init = y_predicted * mask

                # compute first layer relevance according to prediction
                # feature_lrp_scores = lrp_network.lrp(r_init)                 # as Eq(56)
                feature_lrp_scores = lrp_network.lrp(r_init, 'epsilon', 0.01)  # as Eq(58)
                # feature_lrp_scores = nn.lrp(r_init,'alphabeta',2)            # as Eq(60)
                """
                feature_lrp_scores = self.lrp_scores(mlp_network, [x_test[j]])
                avg_feature_lrp_scores = [x + y for x, y in zip(avg_feature_lrp_scores, feature_lrp_scores)]

        if lrp_iterations != 0:
            avg_feature_lrp_scores[:] = [x / lrp_iterations for x in avg_feature_lrp_scores]

        return avg_feature_lrp_scores

    def lrp_scores(self, network, data):
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
                layer_relevance[node_n] = 0

                # calculate relevance input per connected node in higher layer
                for connection_n in range(0, len(weight_matrix[node_n])):
                    if weight_matrix[node_n][connection_n] != 0:
                        # j = node in lower layer; k = node in higher layer
                        effect_j_on_k = activation_matrix[layer_n - 1][0][node_n] * weight_matrix[node_n][connection_n]
                        # calculate the excitatory effects on node k in upper layer
                        sum_effects_on_k = 0
                        for i in range(0, no_of_nodes):
                            sum_effects_on_k += activation_matrix[layer_n - 1][0][i] * weight_matrix[i, connection_n]
                        # calculate the relevance of node j
                        if effect_j_on_k != 0:
                            layer_relevance[node_n] += relevance_matrix[0][connection_n] * (effect_j_on_k / sum_effects_on_k)

            relevance_matrix.insert(0, layer_relevance)

        return relevance_matrix[0]


if __name__ == "__main__":
    X = [(3, 4, 5, 6, 7), (4.2, 5.3, 3, 3, 3), (4, 3, 5, 6, 7), (6, 5, 3, 3, 3),
         (4, 6, 3, 3, 3), (3.7, 5.8, 3, 3, 3), (3.2, 4.6, 3, 3, 3), (5.2, 5.9, 3, 3, 3),
         (5, 4, 3, 3, 3), (7, 4, 3, 3, 3), (3, 7, 3, 3, 3), (4.3, 4.3, 3, 3, 3),
         (-3, -4, 3, 3, 3), (-2, -3.5, 3, 3, 3), (-1, -6, 3, 3, 3), (-3, -4.3, 3, 3, 3),
         (-4, -5.6, 3, 3, 3), (-3.2, -4.8, 3, 3, 3), (-2.3, -4.3, 3, 3, 3), (-2.7, -2.6, 3, 3, 3),
         (-1.5, -3.6, 3, 3, 3), (-3.6, -5.6, 3, 3, 3), (-4.5, -4.6, 3, 3, 3), (-3.7, -5.8, 3, 3, 3)]
    Y = [3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    lrp_nn = LRPNetwork(hidden_layer_sizes=(5, 3), learning_rate_init=0.001, no_of_in_nodes=5)
    avg_lrp_scores = lrp_nn.avg_lrp_score_per_feature(X, Y, test_size=0.1, seed=7)

    print("Average LRP Scores per Feature:")
    print(avg_lrp_scores)
