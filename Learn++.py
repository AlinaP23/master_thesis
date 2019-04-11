"""
Source: https://www.python-course.eu/neural_networks_with_python_numpy.php
Learn++: https://www.researchgate.net/profile/Robi_Polikar/publication/4030043_An_Ensemble_of_Classifiers_Approach_for_the_Missing_Feature_Problem/links/004635182ebc2c7955000000/An-Ensemble-of-Classifiers-Approach-for-the-Missing-Feature-Problem.pdf
"""

import numpy as np
from scipy.stats import truncnorm
import QualityMeasures as qm


@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)


activation_function = sigmoid


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


class NeuralNetwork:
    def __init__(self,
                 no_of_out_nodes,
                 no_of_hidden_nodes,
                 learning_rate,
                 feature_selection):
        self.feature_selection = feature_selection
        self.no_of_in_nodes = self.feature_selection.__len__()
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate
        self.create_weight_matrices()

    def create_weight_matrices(self):
        rad = 1 / np.sqrt(self.no_of_in_nodes)
        x = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_in_hidden = x.rvs((self.no_of_hidden_nodes, self.no_of_in_nodes))
        rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        x = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_out = x.rvs((self.no_of_out_nodes, self.no_of_hidden_nodes))

    def train(self, input_vector, target_vector):
        # input_vector and target_vector can be tuple, list or ndarray
        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T

        output_vector1 = np.dot(self.weights_in_hidden, input_vector)
        output_vector_hidden = activation_function(output_vector1)

        output_vector2 = np.dot(self.weights_hidden_out, output_vector_hidden)
        output_vector_network = activation_function(output_vector2)

        output_errors = target_vector - output_vector_network
        # update the weights:
        tmp = output_errors * output_vector_network * (1.0 - output_vector_network)
        tmp = self.learning_rate * np.dot(tmp, output_vector_hidden.T)
        self.weights_hidden_out += tmp
        # calculate hidden errors:
        hidden_errors = np.dot(self.weights_hidden_out.T, output_errors)
        # update the weights:
        tmp = hidden_errors * output_vector_hidden * (1.0 - output_vector_hidden)
        self.weights_in_hidden += self.learning_rate * np.dot(tmp, input_vector.T)

    def run(self, input_vector):
        """
        running the network with an input vector input_vector.
        input_vector can be tuple, list or ndarray
        """
        # turning the input vector into a column vector
        input_vector = np.array(input_vector, ndmin=2).T
        output_vector = np.dot(self.weights_in_hidden, input_vector)
        output_vector = activation_function(output_vector)

        output_vector = np.dot(self.weights_hidden_out, output_vector)

        return output_vector


class LearnCommittee:
    def __init__(self,
                 no_of_weak_classifiers,
                 percentage_of_features,
                 no_of_features,
                 no_of_out_nodes,
                 no_of_hidden_nodes,
                 learning_rate,
                 labels,
                 p_features,
                 missing_data_representation):
        # to be forwarded to weak classifiers
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate

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

    def train(self,
              data_set):
        size_of_learn_sample = int(len(data_set) * 0.9)
        learn_data = data_set[:size_of_learn_sample]
        test_data = data_set[-size_of_learn_sample:]
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
            weak_classifier = NeuralNetwork(feature_selection=feature_selection,
                                            no_of_hidden_nodes=self.no_of_hidden_nodes,
                                            no_of_out_nodes=self.no_of_out_nodes,
                                            learning_rate=self.learning_rate)

            # train classifier
            for i in range(size_of_learn_sample):
                point = [None] * no_selected_features
                for j in range(0, no_selected_features):
                    point[j] = learn_data[i][0][feature_selection[j]]
                label = learn_data[i][1]
                weak_classifier.train(point, label)

            # calculate classifier quality
            correct_predictions = 0
            false_predictions = 0
            for i in range(len(test_data)):
                point = [None] * no_selected_features
                for j in range(0, no_selected_features):
                    point[j] = test_data[i][0][feature_selection[j]]
                label = test_data[i][1]
                confidence_scores = weak_classifier.run(point)
                max_confidence_label = self.labels[np.argmax(confidence_scores)]
                if max_confidence_label == label:
                    correct_predictions += 1
                else:
                    false_predictions += 1

            # if quality above threshold: save, else discard
            accuracy = correct_predictions / len(test_data)
            if accuracy > 0.5:
                self.universal_classifier_set[k] = weak_classifier
                k += 1
                for i in feature_selection:
                    self.p_features[i] = self.p_features[i] * 1 / self.no_of_features

    def run(self,
            point):
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
            reduced_point = [None] * len(c.feature_selection)
            for i in c.feature_selection:
                reduced_point[c.feature_selection.index(i)] = point[i]
            classification_result = c.run(reduced_point)
            for i in range(classification_result.size):
                summed_up_results[i] += classification_result[i][0]

        # determine weighted majority vote result
        maj_vote_result = summed_up_results.index(max(summed_up_results))

        return maj_vote_result, self.labels[maj_vote_result]


if __name__ == "__main__":
    data1 = [((3, 4, 5, 6, 7), (0.99, 0.01)), ((4.2, 5.3, 3, 3, 3), (0.99, 0.01)),
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

    # standard
    learn_committee = LearnCommittee(no_of_weak_classifiers=20,
                                     percentage_of_features=0.5,
                                     no_of_features=5,
                                     no_of_out_nodes=2,
                                     no_of_hidden_nodes=10,
                                     learning_rate=0.1,
                                     labels=[(0.01, 0.99), (0.99, 0.01)],
                                     missing_data_representation=None,
                                     p_features=None)

    learn_committee.train(data)
    print(learn_committee.run((-4.5, -4.6, None, 3, None)))

    # with LRP probabilities
    learn_committee_LRP = LearnCommittee(no_of_weak_classifiers=20,
                                         percentage_of_features=0.5,
                                         no_of_features=5,
                                         no_of_out_nodes=2,
                                         no_of_hidden_nodes=10,
                                         learning_rate=0.1,
                                         labels=[(0.01, 0.99), (0.99, 0.01)],
                                         missing_data_representation=None,
                                         p_features=[0.2, 0.1, 0.2, 0.2, 0.3])

    learn_committee_LRP.train(data)
    print(learn_committee_LRP.run((-4.5, -4.6, None, 3, None)))
