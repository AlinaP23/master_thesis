"""
Source: https://www.python-course.eu/neural_networks_with_python_numpy.php
Learn++: https://www.researchgate.net/profile/Robi_Polikar/publication/4030043_An_Ensemble_of_Classifiers_Approach_for_the_Missing_Feature_Problem/links/004635182ebc2c7955000000/An-Ensemble-of-Classifiers-Approach-for-the-Missing-Feature-Problem.pdf
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection
from sklearn.metrics import classification_report, accuracy_score


class LearnPlusMLPClassifier(MLPClassifier):
    def __init__(self, feature_selection, hidden_layer_sizes, learning_rate_init):
        self.feature_selection = feature_selection
        super().__init__(hidden_layer_sizes=hidden_layer_sizes,
                         learning_rate_init=learning_rate_init)


class LearnCommittee:
    def __init__(self,
                 no_of_weak_classifiers,
                 percentage_of_features,
                 no_of_features,
                 no_of_out_nodes,
                 hidden_layer_sizes,
                 learning_rate_init,
                 labels,
                 p_features,
                 missing_data_representation):

        # to be forwarded to weak classifiers
        self.no_of_out_nodes = no_of_out_nodes
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init

        # used for LearnCommittee
        self.no_of_features = no_of_features
        self.no_of_weak_classifiers = no_of_weak_classifiers
        self.missing_data_representation = missing_data_representation
        self.percentage_of_features = percentage_of_features
        self.labels = labels
        self.universal_classifier_set = [None]*no_of_weak_classifiers

        # initialize prob. of features to be chosen for weak classifier
        if p_features is not None:
            self.p_features = p_features
        else:
            self.p_features = [None] * self.no_of_features
            for k in range(0, self.no_of_features):
                self.p_features[k] = 1 / self.no_of_features

    def fit(self, features, labels):
        x_weak_train, x_weak_test, y_weak_train, y_weak_test = \
            model_selection.train_test_split(features, labels, test_size=0.1, random_state=7)
        no_selected_features = int(self.no_of_features * self.percentage_of_features)
        feature_range = range(0, self.no_of_features)

        # Training of set of weak classifiers
        k = 0
        while k < self.no_of_weak_classifiers:
            # normalize probability of feature selection
            p_sum = sum(self.p_features)
            if p_sum != 1:
                self.p_features[:] = [x / p_sum for x in self.p_features]

            # random feature selection
            feature_selection = np.random.choice(feature_range, no_selected_features, replace=False, p=self.p_features)\
                .tolist()
            feature_selection.sort()

            # instantiate weak classifier
            weak_classifier = LearnPlusMLPClassifier(feature_selection=feature_selection,
                                                     hidden_layer_sizes=self.hidden_layer_sizes,
                                                     learning_rate_init=self.learning_rate_init)

            # train classifier
            x_reduced = x_weak_train[:, feature_selection]
            weak_classifier.fit(x_reduced, y_weak_train)

            # calculate classifier quality
            y_weak_predicted = weak_classifier.predict(x_weak_test[:, feature_selection])
            accuracy = accuracy_score(y_weak_test, y_weak_predicted)

            # if quality above threshold: save, else discard
            if accuracy > 0.5:
                self.universal_classifier_set[k] = weak_classifier
                k += 1
                for i in feature_selection:
                    self.p_features[i] = self.p_features[i] * 1 / self.no_of_features

    def predict(self, points):
        y_predicted = [None] * len(points)
        for i in range(0, len(points)):
            y_predicted[i] = self.run(points[i])
        return y_predicted

    def run(self, point):
        # determine available features
        available_features = []
        for i in range(0, len(point)):
            if point[i] != self.missing_data_representation:
                available_features.append(i)
        available_features.sort()

        # determine set of usable classifiers
        usable_classifier_set = []
        for c in self.universal_classifier_set:
            if all(feature in available_features for feature in c.feature_selection):
                usable_classifier_set.append(c)

        # classify point with all usable classifiers
        summed_up_results = [0] * self.no_of_out_nodes
        for c in usable_classifier_set:
            reduced_point = point[c.feature_selection].reshape(1, -1)
            classification_result = c.predict_proba(reduced_point)
            summed_up_results = [x + y for x, y in zip(summed_up_results, classification_result[0])]

        # determine weighted majority vote result
        maj_vote_result = summed_up_results.index(max(summed_up_results))

        return maj_vote_result


if __name__ == "__main__":
    iris = pd.read_csv('./data/iris.csv')

    # Create numeric classes for species (0,1,2)
    iris.loc[iris['species'] == 'virginica', 'species'] = 0
    iris.loc[iris['species'] == 'versicolor', 'species'] = 1
    iris.loc[iris['species'] == 'setosa', 'species'] = 2

    # Create Input and Output columns
    X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    Y = iris[['species']].values.ravel()

    x_train, x_test, y_train, y_test = \
        model_selection.train_test_split(X, Y, test_size=0.1, random_state=7)

    # standard
    learn_committee = LearnCommittee(no_of_weak_classifiers=20,
                                     percentage_of_features=0.5,
                                     no_of_features=4,
                                     no_of_out_nodes=3,
                                     hidden_layer_sizes=(10, 10, 10),
                                     learning_rate_init=0.01,
                                     labels=[2, 1, 0],
                                     missing_data_representation=None,
                                     p_features=None)
    learn_committee.fit(x_train, y_train)

    # LRP
    learn_committee_lrp = LearnCommittee(no_of_weak_classifiers=20,
                                         percentage_of_features=0.5,
                                         no_of_features=4,
                                         no_of_out_nodes=3,
                                         hidden_layer_sizes=(10, 10, 10),
                                         learning_rate_init=0.01,
                                         labels=[2, 1, 0],
                                         missing_data_representation=None,
                                         p_features=[0.4, 0.1, 0.1, 0.4])
    learn_committee_lrp.fit(x_train, y_train)

    # simulate random sensor failure
    features = range(0, len(x_test[0]))
    p_failure = [1/len(x_test[0])] * len(x_test[0])
    x_test_failure = np.copy(x_test)

    for i in range(0, len(x_test)):
        sensor_failure = np.random.choice(features, 1, replace=False, p=p_failure).tolist()
        x_test_failure[i, sensor_failure] = 0

    print("Accuracy Score - Learn++:")
    predictions = learn_committee.predict(x_test)
    print("w/o LRP & w/o Sensor Failure: ", accuracy_score(predictions, y_test))

    predictions_failure = learn_committee.predict(x_test_failure)
    print("w/o LRP & w/  Sensor Failure: ", accuracy_score(predictions_failure, y_test))

    predictions_lrp = learn_committee_lrp.predict(x_test)
    print("w/ LRP  & w/o Sensor Failure: ", accuracy_score(predictions_lrp, y_test))

    predictions_failure_lrp = learn_committee_lrp.predict(x_test_failure)
    print("w/ LRP  & w/  Sensor Failure: ", accuracy_score(predictions_failure_lrp, y_test))
