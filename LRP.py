import numpy as np
from scipy.stats import truncnorm
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import lrp_toolbox.python.model_io as model_io
from lrp_toolbox.python.modules import Sequential

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

        # save the model to disk
        filename = 'lrp_network.nn'
        pickle.dump(mlp_network, open(filename, 'wb'))

        predictions = mlp_network.predict(x_test)
        print(classification_report(y_test, predictions))

        # calculate avg. LRP scores for features - use correctly classified test data to determine LRP scores
        lrp_iterations = 0
        lrp_network = model_io.read(filename)

        for i in range(0, len(y_test)):
            if y_test[i] == predictions[i]:
                lrp_iterations += 1
                y_predicted = lrp_network.predict_proba([x_test[i]])

                # prepare initial relevance to reflect the model's dominant prediction
                # (ie depopulate non-dominant output neurons)
                mask = np.zeros_like(y_predicted)
                mask[:, np.argmax(y_predicted)] = 1
                r_init = y_predicted * mask

                # compute first layer relevance according to prediction
                # feature_lrp_scores = lrp_network.lrp(r_init)                 # as Eq(56)
                feature_lrp_scores = lrp_network.lrp(r_init, 'epsilon', 0.01)  # as Eq(58)
                # feature_lrp_scores = nn.lrp(r_init,'alphabeta',2)            # as Eq(60)

                avg_feature_lrp_scores += feature_lrp_scores

        avg_feature_lrp_scores[:] = [x / lrp_iterations for x in avg_feature_lrp_scores]

        return avg_feature_lrp_scores


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
