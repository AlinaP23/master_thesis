"""
Source: https://www.python-course.eu/neural_networks_with_python_numpy.php
Learn++: https://www.researchgate.net/profile/Robi_Polikar/publication/4030043_An_Ensemble_of_Classifiers_Approach_for_the_Missing_Feature_Problem/links/004635182ebc2c7955000000/An-Ensemble-of-Classifiers-Approach-for-the-Missing-Feature-Problem.pdf
"""

import numpy as np
from scipy.stats import truncnorm
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
    """data1 = [((3, 4, 5, 6, 7), (0.99, 0.01)), ((4.2, 5.3, 3, 3, 3), (0.99, 0.01)),
             ((4, 3, 5, 6, 7), (0.99, 0.01)), ((6, 5, 3, 3, 3), (0.99, 0.01)),
             ((4, 6, 3, 3, 3), (0.99, 0.01)), ((3.7, 5.8, 3, 3, 3), (0.99, 0.01)),
             ((3.2, 4.6, 3, 3, 3), (0.99, 0.01)), ((5.2, 5.9, 3, 3, 3), (0.99, 0.01)),
             ((5, 4, 3, 3, 3), (0.99, 0.01)), ((7, 4, 3, 3, 3), (0.99, 0.01)),
             ((3, 7, 3, 3, 3), (0.99, 0.01)), ((4.3, 4.3, 3, 3, 3), (0.99, 0.01))]
    data2 = [((-3, -4, 3, 3, 3), (0.01, 0.99)), ((-2, -3.5, 3, 3, 3), (0.01, 0.99)),
             ((-1, -6, 3, 3, 3), (0.01, 0.99)), ((-3, -4.3, 3, 3, 3), (0.01, 0.99)),
             ((-4, -5.6, 3, 3, 3), (0.01, 0.99)), ((-3.2, -4.8, 3, 3, 3), (0.01, 0.99)),
             ((-2.3, -4.3, 3, 3, 3), (0.01, 0.99)), ((-2.7, -2.6, 3, 3, 3), (0.01, 0.99)),
             ((-1.5, -3.6, 3, 3, 3), (0.01, 0.99)), ((-3.6, -5.6, 3, 3, 3), (0.01, 0.99)),
             ((-4.5, -4.6, 3, 3, 3), (0.01, 0.99)), ((-3.7, -5.8, 3, 3, 3), (0.01, 0.99))]
    data = data1 + data2
    np.random.shuffle(data)
    """
    iris = pd.read_csv('./data/iris.csv')

    # Create numeric classes for species (0,1,2)
    iris.loc[iris['species'] == 'virginica', 'species'] = 0
    iris.loc[iris['species'] == 'versicolor', 'species'] = 1
    iris.loc[iris['species'] == 'setosa', 'species'] = 2

    # Create Input and Output columns
    X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    Y = iris[['species']].values.ravel()

    # standard
    learn_committee = LearnCommittee(no_of_weak_classifiers=20,
                                     percentage_of_features=0.5,
                                     no_of_features=4,
                                     no_of_out_nodes=3,
                                     hidden_layer_sizes=(3, 3, 3),
                                     learning_rate_init=0.01,
                                     labels=[2, 1, 0],
                                     missing_data_representation=None,
                                     p_features=None)

    x_train, x_test, y_train, y_test = \
        model_selection.train_test_split(X, Y, test_size=0.1, random_state=7)
    learn_committee.fit(x_train, y_train)

    # simulate random sensor failure
    features = range(0, len(x_test[0]))
    p_failure = [1/len(x_test[0])] * len(x_test[0])
    x_test_failure = np.copy(x_test)

    for i in range(0, len(x_test)):
        sensor_failure = np.random.choice(features, 1, replace=False, p=p_failure).tolist()
        x_test_failure[i, sensor_failure] = 0

    predictions = learn_committee.predict(x_test)
    print("Accuracy Score - Learn++:")
    print(accuracy_score(predictions, y_test))

    predictions_failure = learn_committee.predict(x_test_failure)
    print("Accuracy Score - Learn++ w/ sensor failure:")
    print(accuracy_score(predictions_failure, y_test))
