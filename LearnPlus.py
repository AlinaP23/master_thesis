import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score


class LearnPlusMLPClassifier(MLPClassifier):
    def __init__(self, feature_selection, hidden_layer_sizes, learning_rate_init, activation):
        self.feature_selection = feature_selection
        super().__init__(hidden_layer_sizes=hidden_layer_sizes,
                         learning_rate_init=learning_rate_init,
                         activation=activation)


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
                 missing_data_representation,
                 activation,
                 threshold):

        # to be forwarded to weak classifiers
        self.no_of_out_nodes = no_of_out_nodes
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init
        self.activation = activation
        self.threshold = threshold

        # used for LearnCommittee
        self.no_of_features = no_of_features
        self.no_of_weak_classifiers = no_of_weak_classifiers
        self.missing_data_representation = missing_data_representation
        self.percentage_of_features = percentage_of_features
        self.labels = labels
        self.universal_classifier_set = [None]*no_of_weak_classifiers

        # initialize probability of features to be chosen for weak classifier
        if p_features is not None:
            self.p_features = np.copy(p_features)
        else:
            self.p_features = [None] * self.no_of_features
            for k in range(0, self.no_of_features):
                self.p_features[k] = 1 / self.no_of_features

    def fit(self, features, labels, np_seed, split_seed):
        """Triggers the training of the Learn++-Committee.

        Parameters
        ----------
        features: array of shape [n_samples, n_features]
            Samples to be used for training of the committee
        labels: array of shape [n_samples]
            Labels for class membership of each sample
        np_seed: integer
            Seed to make numpy randomization reproducible.
        split_seed: integer
            Seed to make random split reproducible.
        """
        x_weak_train, x_weak_test, y_weak_train, y_weak_test = \
            model_selection.train_test_split(features, labels, test_size=0.1, random_state=split_seed, stratify=labels)
        no_selected_features = int(self.no_of_features * self.percentage_of_features)
        feature_range = range(0, self.no_of_features)
        np.random.seed(np_seed)

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
                                                     learning_rate_init=self.learning_rate_init,
                                                     activation=self.activation)

            # train classifier
            x_reduced = x_weak_train[:, feature_selection]
            weak_classifier.fit(x_reduced, y_weak_train)

            # calculate classifier quality
            y_weak_predicted = weak_classifier.predict(x_weak_test[:, feature_selection])
            accuracy = accuracy_score(y_weak_test, y_weak_predicted)

            # if quality above threshold: save, else discard
            if accuracy > self.threshold:
                self.universal_classifier_set[k] = weak_classifier
                k += 1
                print(k, " weak classifiers trained")
                for i in feature_selection:
                    self.p_features[i] = self.p_features[i] * 1 / self.no_of_features

    def predict(self, points, data_frame=False):
        """Classify the given input using the Learn++-Committee.

        Parameters
        ----------
        points: array of shape [n_samples, n_features]
            Samples to be classified
        data_frame: boolean
            Indicates whether the label array to be returned should be transformed to a data frame

        Returns
        ----------
        y_predicted: array of shape [n_samples]
            Predicted labels for the given points
        """
        y_predicted = [None] * len(points)
        for p in range(0, len(points)):
            y_predicted[p] = self.labels[self.run(points[p])]
        if data_frame:
            y_predicted = pd.DataFrame(list(y_predicted))
        return y_predicted

    def run(self, point):
        """Classify the a single sample via majority vote of the committee's weak classifiers.

         Parameters
         ----------
         point: array of shape [n_features]
            Sample to be classified

         Returns
         ----------
         y_predicted: integer
            Index of the predicted label for the given point
         """
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

