"""
Source: https://www.python-course.eu/neural_networks_with_python_numpy.php
DropIn: https://arxiv.org/pdf/1705.02643.pdf
"""

from scipy.stats import truncnorm
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.utils.extmath import safe_sparse_dot
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
            for j in range(len(activations[0])):
                # DropIn implementation
                if type(self.p_dropin) is list:
                    self.dropout_arrays[j] = np.random.binomial(1, self.p_dropin)
                else:
                    self.dropout_arrays[j] = np.random.binomial(1, self.p_dropin, size=activations[0][j].shape)
                activations[0][j] = activations[0][j] * self.dropout_arrays[j]

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


if __name__ == "__main__":
    data_set = "bank"

    if data_set == "iris":
        iris = pd.read_csv('./data/iris.csv')

        # Create numeric classes for species (0,1,2)
        iris.loc[iris['species'] == 'virginica', 'species'] = 0
        iris.loc[iris['species'] == 'versicolor', 'species'] = 1
        iris.loc[iris['species'] == 'setosa', 'species'] = 2

        # Create Input and Output columns
        X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
        Y = iris[['species']].values.ravel()

    elif data_set == "bank":
        bank = pd.read_csv('./data/bank_data.csv')

        # Create Input and Output columns
        X = bank[['age', 'job_num', 'marital_num', 'education_num', 'default_num', 'housing_num', 'loan_num	',
                  'contact_num', 'month_num', 'day_num', 'duration', 'campaign', 'pdays', 'previous', 'poutcome',
                  'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']].values
        Y = bank[['y']].values.ravel()

    x_train, x_test, y_train, y_test = \
        model_selection.train_test_split(X, Y, test_size=0.1, random_state=7)

    # NEURAL NETWORKS PARAMETERS
    hidden_layer_sizes = [10, 10, 10]
    learning_rate_init = 0.1
    p_dropin_standard = 0.9
    p_dropin_lrp = [0.1, 0.46, 0.19, 0.59]
    p_dropin_lrp_range = [0.1, 0.69, 0.24, 0.9]

    # standard
    dropin_network = DropInNetwork(hidden_layer_sizes=hidden_layer_sizes,
                                   learning_rate_init=learning_rate_init,
                                   p_dropin=p_dropin_standard)
    dropin_network.fit_dropin(x_train, y_train)

    # LRP
    dropin_network_lrp = DropInNetwork(hidden_layer_sizes=hidden_layer_sizes,
                                       learning_rate_init=learning_rate_init,
                                       p_dropin=p_dropin_lrp)
    dropin_network_lrp.fit_dropin(x_train, y_train)

    # LRP - Range
    dropin_network_lrp_r = DropInNetwork(hidden_layer_sizes=hidden_layer_sizes,
                                         learning_rate_init=learning_rate_init,
                                         p_dropin=p_dropin_lrp_range)
    dropin_network_lrp_r.fit_dropin(x_train, y_train)

    # simulate random sensor failure
    features = range(0, len(x_test[0]))
    p_failure = [1/len(x_test[0])] * len(x_test[0])
    x_test_failure = np.copy(x_test)

    for i in range(0, len(x_test)):
        sensor_failure = np.random.choice(features, 1, replace=False, p=p_failure).tolist()
        x_test_failure[i, sensor_failure] = 0

    print("Accuracy Score - DropIn:")
    predictions = dropin_network.predict(x_test)
    print("w/o LRP  & w/o Sensor Failure: ", accuracy_score(predictions, y_test))

    predictions_failure = dropin_network.predict(x_test_failure)
    print("w/o LRP  & w/  Sensor Failure: ", accuracy_score(predictions_failure, y_test))

    predictions_lrp = dropin_network_lrp.predict(x_test)
    print("w/  LRP  & w/o Sensor Failure: ", accuracy_score(predictions_lrp, y_test))

    predictions_failure_lrp = dropin_network_lrp.predict(x_test_failure)
    print("w/  LRP  & w/  Sensor Failure: ", accuracy_score(predictions_failure_lrp, y_test))

    predictions_lrp_r = dropin_network_lrp_r.predict(x_test)
    print("w/  LRPr & w/o Sensor Failure: ", accuracy_score(predictions_lrp_r, y_test))

    predictions_failure_lrp_r = dropin_network_lrp_r.predict(x_test_failure)
    print("w/  LRPr & w/  Sensor Failure: ", accuracy_score(predictions_failure_lrp_r, y_test))
