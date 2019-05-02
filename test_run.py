import data_lib
from LRP import LRPNetwork
from LearnPlus_scikit import LearnCommittee
from DropIn_scikit import DropInNetwork
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import numpy as np

# --- PARAMETERS --- #
# General
X, Y, activation, labels = data_lib.get_dataset("income")
x_train, x_test, y_train, y_test = \
    model_selection.train_test_split(X, Y, test_size=0.1, random_state=7)
x_test_failure = data_lib.get_sensor_failure_test_set(x_test, random=True)

# LRP
LRP_hidden_layer_sizes = (5, 6, 5)
LRP_learning_rate_init = 0.1
LRP_test_size = 0.2
LRP_seed = 7
LRP_alpha = 1
LRP_accuracy_threshold = 0.7
LRP_iterations = 10
LRP_dropout_threshold_max = 0.9
LRP_dropout_threshold_min = 0.2

# Learn++
learn_no_of_weak_classifiers = 20
learn_percentage_of_features = 0.75
learn_no_of_features = 11
learn_no_of_out_nodes = 2
learn_hidden_layer_sizes = [10, 10, 10]
learn_learning_rate_init = 0.1
learn_missing_data_representation = None
learn_p_features_standard = None

# DropIn
dropin_hidden_layer_sizes = [10, 10, 10]
dropin_learning_rate_init = 0.1
p_dropin_standard = 0.8

# --- LRP Score Calculation --- #
lrp_nn = LRPNetwork(hidden_layer_sizes=LRP_hidden_layer_sizes,
                    learning_rate_init=LRP_learning_rate_init,
                    no_of_in_nodes=len(X[0]),
                    activation=activation)

avg_lrp_scores = lrp_nn.avg_lrp_score_per_feature(features=X,
                                                  labels=Y,
                                                  test_size=LRP_test_size,
                                                  seed=LRP_seed,
                                                  alpha=LRP_alpha,
                                                  accuracy_threshold=LRP_accuracy_threshold,
                                                  iterations=LRP_iterations)

avg_lrp_scores_normalized = lrp_nn.lrp_scores_to_percentage(avg_lrp_scores)
avg_lrp_scores_scaled, avg_lrp_scores_scaled_inverted = \
    lrp_nn.lrp_scores_to_scaled(avg_lrp_scores, LRP_dropout_threshold_max)
avg_lrp_scores_range, avg_lrp_scores_range_inverted = \
    lrp_nn.lrp_scores_to_scaled(avg_lrp_scores, LRP_dropout_threshold_max, LRP_dropout_threshold_min)

print("Number of tuples taken into consideration:")
print(lrp_nn.LRP_scores_regarded)
print("Average LRP Scores per Feature:")
print(avg_lrp_scores)
print("Normalized - to be used for Learn++:")
print(avg_lrp_scores_normalized)
print("Scaled to Dropout probabilities:")
print(avg_lrp_scores_scaled)
print("Inverted to Dropin probabilities - to be used for DropIn:")
print(avg_lrp_scores_scaled_inverted)
print("Scaled by range to Dropout prob.:")
print(avg_lrp_scores_range)
print("Inverted range prob.:")
print(avg_lrp_scores_range_inverted)

# --- Learn++ --- #
# standard
learn_committee = LearnCommittee(no_of_weak_classifiers=learn_no_of_weak_classifiers,
                                 percentage_of_features=learn_percentage_of_features,
                                 no_of_features=learn_no_of_features,
                                 no_of_out_nodes=learn_no_of_out_nodes,
                                 hidden_layer_sizes=learn_hidden_layer_sizes,
                                 learning_rate_init=learn_learning_rate_init,
                                 labels=labels,
                                 missing_data_representation=learn_missing_data_representation,
                                 p_features=learn_p_features_standard,
                                 activation=activation)
learn_committee.fit(x_train, y_train)

# LRP
learn_committee_lrp = LearnCommittee(no_of_weak_classifiers=learn_no_of_weak_classifiers,
                                     percentage_of_features=learn_percentage_of_features,
                                     no_of_features=learn_no_of_features,
                                     no_of_out_nodes=learn_no_of_out_nodes,
                                     hidden_layer_sizes=learn_hidden_layer_sizes,
                                     learning_rate_init=learn_learning_rate_init,
                                     labels=labels,
                                     missing_data_representation=learn_missing_data_representation,
                                     p_features=avg_lrp_scores_normalized,
                                     activation=activation)
learn_committee_lrp.fit(x_train, y_train)

# Validation
print("Accuracy Score - Learn++:")
predictions = learn_committee.predict(x_test)
print("w/o LRP & w/o Sensor Failure: ", accuracy_score(predictions, y_test))

predictions_failure = learn_committee.predict(x_test_failure)
print("w/o LRP & w/  Sensor Failure: ", accuracy_score(predictions_failure, y_test))

predictions_lrp = learn_committee_lrp.predict(x_test)
print("w/ LRP  & w/o Sensor Failure: ", accuracy_score(predictions_lrp, y_test))

predictions_failure_lrp = learn_committee_lrp.predict(x_test_failure)
print("w/ LRP  & w/  Sensor Failure: ", accuracy_score(predictions_failure_lrp, y_test))

# --- DropIn --- #
# Standard
dropin_network = DropInNetwork(hidden_layer_sizes=dropin_hidden_layer_sizes,
                               learning_rate_init=dropin_learning_rate_init,
                               p_dropin=p_dropin_standard)
dropin_network.fit_dropin(x_train, y_train)

# LRP
dropin_network_lrp = DropInNetwork(hidden_layer_sizes=dropin_hidden_layer_sizes,
                                   learning_rate_init=dropin_learning_rate_init,
                                   p_dropin=avg_lrp_scores_scaled_inverted)
dropin_network_lrp.fit_dropin(x_train, y_train)

# LRP - Range
dropin_network_lrp_r = DropInNetwork(hidden_layer_sizes=dropin_hidden_layer_sizes,
                                     learning_rate_init=dropin_learning_rate_init,
                                     p_dropin=avg_lrp_scores_range_inverted)
dropin_network_lrp_r.fit_dropin(x_train, y_train)

# Validation
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
