import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import lrp_toolbox.python.model_io as model_io
from lrp_toolbox.python.modules import Sequential
from lrp_toolbox.python.modules import Rect


class LRPNetwork:
    def __init__(self,
                 hidden_layer_sizes,
                 learning_rate,
                 no_of_in_nodes):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.no_of_in_nodes = no_of_in_nodes

    def avg_lrp_score_per_feature(self, features, labels, test_size, seed):
        # variable definition
        avg_feature_lrp_scores = [0] * self.no_of_in_nodes

        x_train, x_test, y_train, y_test = \
            model_selection.train_test_split(features, labels, test_size=test_size, random_state=seed)
        # train neural network
        mlp_network = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes,
                                    learning_rate_init=self.learning_rate)
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
                feature_lrp_scores = self.lrp_scores(mlp_network, x_test[j])
                avg_feature_lrp_scores += feature_lrp_scores

        avg_feature_lrp_scores[:] = [x / lrp_iterations for x in avg_feature_lrp_scores]

        return avg_feature_lrp_scores

    def lrp_scores(self, network, data):
        # TODO: alter predict function to return probabilities of output layer along with a matrix containing the node activations (m=no_layers, n=no_nodes)
        y_predicted, activation_matrix = network.predict_proba(data)
        relevance_matrix = list
        relevance_matrix.append(y_predicted)
        # loop over layers
        for layer_n in range(network.n_layers_, 0, -1):
            no_of_nodes = self.hidden_layer_sizes[layer_n - 1]
            weight_matrix = network.coefs_[layer_n - 1]
            layer_relevance = [0] * no_of_nodes
            # loop over nodes
            for node_n in range(0, no_of_nodes):
                layer_relevance[node_n] = 0
                for connection_n in range(0, len(weight_matrix[node_n])):
                    if weight_matrix[node_n][connection_n] != 0:
                        # j = node in lower layer; k = node in higher layer
                        effect_j_on_k = activation_matrix[layer_n][node_n] * weight_matrix[node_n][connection_n]
                        # calculate the excitatory effects on node k in upper layer
                        sum_effects_on_k = 0
                        for i in range(0, no_of_nodes):
                            sum_effects_on_k += activation_matrix[layer_n][i] * weight_matrix[i][connection_n]
                        # calculate the relevance of node j
                        layer_relevance[node_n] += relevance_matrix[0][connection_n] * (effect_j_on_k / sum_effects_on_k)

            relevance_matrix.insert(index=0, object=layer_relevance)

        return relevance_matrix[0]

if __name__ == "__main__":
    X = [(3, 4, 5, 6, 7), (4.2, 5.3, 3, 3, 3), (4, 3, 5, 6, 7), (6, 5, 3, 3, 3),
         (4, 6, 3, 3, 3), (3.7, 5.8, 3, 3, 3), (3.2, 4.6, 3, 3, 3), (5.2, 5.9, 3, 3, 3),
         (5, 4, 3, 3, 3), (7, 4, 3, 3, 3), (3, 7, 3, 3, 3), (4.3, 4.3, 3, 3, 3),
         (-3, -4, 3, 3, 3), (-2, -3.5, 3, 3, 3), (-1, -6, 3, 3, 3), (-3, -4.3, 3, 3, 3),
         (-4, -5.6, 3, 3, 3), (-3.2, -4.8, 3, 3, 3), (-2.3, -4.3, 3, 3, 3), (-2.7, -2.6, 3, 3, 3),
         (-1.5, -3.6, 3, 3, 3), (-3.6, -5.6, 3, 3, 3), (-4.5, -4.6, 3, 3, 3), (-3.7, -5.8, 3, 3, 3)]
    Y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    lrp_nn = LRPNetwork(hidden_layer_sizes=(5, 3), learning_rate=0.1, no_of_in_nodes=5)
    avg_lrp_scores = lrp_nn.avg_lrp_score_per_feature(X, Y, test_size=0.1, seed=7)

    print("Average LRP Scores per Feature:")
    print(avg_lrp_scores)
