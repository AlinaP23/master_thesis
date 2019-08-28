"""
Source: https://www.python-course.eu/neural_networks_with_python_numpy.php
Learn++: https://www.researchgate.net/profile/Robi_Polikar/publication/4030043_An_Ensemble_of_Classifiers_Approach_for_the_Missing_Feature_Problem/links/004635182ebc2c7955000000/An-Ensemble-of-Classifiers-Approach-for-the-Missing-Feature-Problem.pdf
"""
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn import model_selection
from sklearn.metrics import r2_score


class LearnPlusMLPRegressor(MLPRegressor):
    def __init__(self, feature_selection, hidden_layer_sizes, learning_rate_init, activation):
        self.feature_selection = feature_selection
        super().__init__(hidden_layer_sizes=hidden_layer_sizes,
                         learning_rate_init=learning_rate_init,
                         activation=activation)


class LearnCommitteeRegression:
    def __init__(self,
                 no_of_weak_regressors,
                 percentage_of_features,
                 no_of_features,
                 hidden_layer_sizes,
                 learning_rate_init,
                 p_features,
                 missing_data_representation,
                 activation,
                 threshold):

        # to be forwarded to weak regressors
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init
        self.activation = activation
        self.threshold = threshold

        # used for LearnCommittee
        self.no_of_features = no_of_features
        self.no_of_weak_regressors = no_of_weak_regressors
        self.missing_data_representation = missing_data_representation
        self.percentage_of_features = percentage_of_features
        self.universal_regressor_set = [None]*no_of_weak_regressors

        # initialize prob. of features to be chosen for weak regressors
        if p_features is not None:
            self.p_features = np.copy(p_features)
        else:
            self.p_features = [None] * self.no_of_features
            for k in range(0, self.no_of_features):
                self.p_features[k] = 1 / self.no_of_features

    def fit(self, features, target_values, np_seed, split_seed):
        """Triggers the training of the Learn++-Committee.

        Parameters
        ----------
        features: array of shape [n_samples, n_features]
            Samples to be used for training of the committee
        target_values: array of shape [n_samples]
            Target values of the samples
        np_seed: integer
            Seed to make numpy randomization reproducible.
        split_seed: integer
            Seed to make random split reproducible.
        """
        x_weak_train, x_weak_test, y_weak_train, y_weak_test = \
            model_selection.train_test_split(features, target_values, test_size=0.1, random_state=split_seed)
        no_selected_features = int(self.no_of_features * self.percentage_of_features)
        feature_range = range(0, self.no_of_features)
        np.random.seed(np_seed)

        # Training of set of weak regressors
        k = 0
        while k < self.no_of_weak_regressors:
            # normalize probability of feature selection
            p_sum = sum(self.p_features)
            if p_sum != 1:
                self.p_features[:] = [x / p_sum for x in self.p_features]

            # random feature selection
            feature_selection = np.random.choice(feature_range, no_selected_features, replace=False, p=self.p_features)\
                .tolist()
            feature_selection.sort()

            # instantiate weak regressor
            weak_regressor = LearnPlusMLPRegressor(feature_selection=feature_selection,
                                                   hidden_layer_sizes=self.hidden_layer_sizes,
                                                   learning_rate_init=self.learning_rate_init,
                                                   activation=self.activation)

            # train regressor
            x_reduced = x_weak_train[:, feature_selection]
            weak_regressor.fit(x_reduced, y_weak_train)

            # calculate regressor quality
            y_weak_predicted = weak_regressor.predict(x_weak_test[:, feature_selection])
            r_2 = r2_score(y_weak_test, y_weak_predicted)

            # if quality above threshold: save, else discard
            if abs(r_2) < self.threshold:
                self.universal_regressor_set[k] = weak_regressor
                k += 1
                print(k, " weak regressors trained")
                for i in feature_selection:
                    self.p_features[i] = self.p_features[i] * 1 / self.no_of_features

    def predict(self, points):
        """Predict target variable for the given input using the Learn++-Committee.

        Parameters
        ----------
        points: array of shape [n_samples, n_features]
            Samples to be classified

        Returns
        ----------
        y_predicted: array of shape [n_samples]
            Predicted values for the given points
        """
        y_predicted = [None] * len(points)
        for p in range(0, len(points)):
            y_predicted[p] = self.run(points[p])

        return y_predicted

    def run(self, point):
        """Predict the target value of a single sample by averaging over the committee's weak regressors' results.

         Parameters
         ----------
         point: array of shape [n_features]
            Sample for which target value shall be predicted

         Returns
         ----------
         y_predicted: integer
            Committee's final prediction (average of the individual regressors' results)
         """
        # determine available features
        available_features = []
        for i in range(0, len(point)):
            if point[i] != self.missing_data_representation:
                available_features.append(i)
        available_features.sort()

        # determine set of usable regressors
        usable_regressor_set = []
        for c in self.universal_regressor_set:
            if all(feature in available_features for feature in c.feature_selection):
                usable_regressor_set.append(c)

        # classify point with all usable regressors
        summed_up_results = [0]
        for c in usable_regressor_set:
            reduced_point = point[c.feature_selection].reshape(1, -1)
            regression_result = c.predict(reduced_point)
            summed_up_results = [x + y for x, y in zip(summed_up_results, regression_result)]

        # determine committee result by averaging the individual results
        avg_result = summed_up_results[0] / len(usable_regressor_set)

        return avg_result

