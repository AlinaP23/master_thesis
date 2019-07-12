import data_lib
from LRP import LRPNetwork
from LearnPlus import LearnCommittee
from DropIn import DropInNetwork
from SelectiveRetraining import SelectiveRetrainingCommittee
from sklearn import model_selection
from sklearn.metrics import accuracy_score

# --- PARAMETERS --- #
# General
algorithms_to_execute = {"LRP":     True,
                         "Learn++": True,
                         "DropIn":  True,
                         "SelectiveRetraining": True}
data_set = "WBC"
data_set_params = {"n_samples":     50000,
                   "n_features":    15,
                   "n_informative": 9,
                   "n_redundant":   3,
                   "n_repeated":    2,
                   "n_classes":     3,
                   "n_clusters_per_class":  2,
                   "weights":       None,
                   "flip_y":        0.01,
                   "class_sep":     1.0,
                   "hypercube":     True,
                   "shift":         0.0,
                   "scale":         1.0,
                   "data_shuffle":  True,
                   "random_state":  7}
ms_random_state = 9
ms_test_size = 0.3
failure_simulation_np_seed = 7
failure_percentages = [0.10, 0.2, 0.25, 0.3, 0.40, 0.55, 0.70, 0.80]
random_failure = False
multi_sensor_failure = True

X, Y, activation, labels, label_df = data_lib.get_data_set(data_set,
                                                           n_samples=data_set_params["n_samples"],
                                                           n_features=data_set_params["n_features"],
                                                           n_informative=data_set_params["n_informative"],
                                                           n_redundant=data_set_params["n_redundant"],
                                                           n_repeated=data_set_params["n_repeated"],
                                                           n_classes=data_set_params["n_classes"],
                                                           n_clusters_per_class=data_set_params["n_clusters_per_class"],
                                                           weights=data_set_params["weights"],
                                                           flip_y=data_set_params["flip_y"],
                                                           class_sep=data_set_params["class_sep"],
                                                           hypercube=data_set_params["hypercube"],
                                                           shift=data_set_params["shift"],
                                                           scale=data_set_params["scale"],
                                                           data_shuffle=data_set_params["data_shuffle"],
                                                           random_state=data_set_params["random_state"])
x_train, x_test, y_train, y_test = \
    model_selection.train_test_split(X, Y, test_size=ms_test_size, random_state=ms_random_state, stratify=Y)

x_test_failures = [data_lib.get_sensor_failure_test_set(x_test,
                                                        np_seed=failure_simulation_np_seed,
                                                        random=random_failure,
                                                        multi_sensor_failure=multi_sensor_failure,
                                                        failure_percentage=p)
                   for p in failure_percentages]
# LRP
LRP_hidden_layer_sizes = [60, 60, 60]
LRP_learning_rate_init = 0.1
LRP_random_states = [7]
LRP_seed = 9
LRP_test_size = 0.1
LRP_alpha = 2
LRP_accuracy_threshold = 0.3
LRP_dropout_threshold_max = 0.4
LRP_dropout_threshold_min = 0.1

# Learn++
learn_hidden_layer_sizes = [30, 30, 30]
learn_learning_rate_init = 0.1
learn_random_state = 9
learn_np_seed = 5
learn_no_of_weak_classifiers = 20
learn_percentage_of_features = [0.3, 0.4, 0.6, 0.8]
learn_missing_data_representation = None
learn_p_features_standard = None
learn_p_weak_classifier_threshold = 0.5

# DropIn
dropin_hidden_layer_sizes = [60, 60, 60]
dropin_learning_rate_init = 0.1
dropin_random_state = 9
dropin_np_seed = 9
p_dropin_standard = [0.8]
dropin_epochs = 3

# Selective Retraining
sr_hidden_layer_sizes = [60, 60, 60]
sr_learning_rate_init = 0.1
sr_random_state = 7
sr_weight_threshold = [0.25, 0.5, 0.75]

# --- LRP Score Calculation --- #
if algorithms_to_execute["LRP"]:
    print("Training LRP Network...")
    lrp_nn = LRPNetwork(hidden_layer_sizes=LRP_hidden_layer_sizes,
                        learning_rate_init=LRP_learning_rate_init,
                        no_of_in_nodes=len(X[0]),
                        activation=activation)
    print("Calculating LRP Scores...")
    avg_lrp_scores, mlp_network = lrp_nn.avg_lrp_score_per_feature(features=X,
                                                                   labels=Y,
                                                                   test_size=LRP_test_size,
                                                                   seed=LRP_seed,
                                                                   random_states=LRP_random_states,
                                                                   alpha=LRP_alpha,
                                                                   accuracy_threshold=LRP_accuracy_threshold)
    print("Validating LRP NN...")
    lrp_predictions = mlp_network.predict(x_test)
    lrp_predictions_failure = []
    for lrp_failure_test in x_test_failures:
        lrp_predictions_failure.append(mlp_network.predict(lrp_failure_test))

    print("Transforming LRP Scores...")
    avg_lrp_scores_normalized, avg_lrp_scores_normalized_inverted = lrp_nn.lrp_scores_to_percentage(avg_lrp_scores)
    avg_lrp_scores_scaled, avg_lrp_scores_scaled_inverted = \
        lrp_nn.lrp_scores_to_scaled(avg_lrp_scores, LRP_dropout_threshold_max)
    avg_lrp_scores_range, avg_lrp_scores_range_inverted = \
        lrp_nn.lrp_scores_to_scaled_range(avg_lrp_scores, LRP_dropout_threshold_max, LRP_dropout_threshold_min)

# --- Learn++ --- #
if algorithms_to_execute["Learn++"]:
    # standard
    print("Training Standard Learn++ Committee...")
    learn_predictions = []
    learn_predictions_lrp = []
    learn_predictions_failure_set = []
    learn_predictions_failure_lrp_set = []
    for pof in learn_percentage_of_features:
        learn_committee = LearnCommittee(no_of_weak_classifiers=learn_no_of_weak_classifiers,
                                         percentage_of_features=pof,
                                         no_of_features=len(X[0]),
                                         no_of_out_nodes=len(labels),
                                         hidden_layer_sizes=learn_hidden_layer_sizes,
                                         learning_rate_init=learn_learning_rate_init,
                                         labels=labels,
                                         missing_data_representation=learn_missing_data_representation,
                                         p_features=learn_p_features_standard,
                                         activation=activation,
                                         threshold=learn_p_weak_classifier_threshold)
        learn_committee.fit(x_train, y_train, learn_np_seed, learn_random_state)

        if algorithms_to_execute["LRP"]:
            # LRP
            print("Training LRP Learn++ Committee...")
            learn_committee_lrp = LearnCommittee(no_of_weak_classifiers=learn_no_of_weak_classifiers,
                                                 percentage_of_features=pof,
                                                 no_of_features=len(X[0]),
                                                 no_of_out_nodes=len(labels),
                                                 hidden_layer_sizes=learn_hidden_layer_sizes,
                                                 learning_rate_init=learn_learning_rate_init,
                                                 labels=labels,
                                                 missing_data_representation=learn_missing_data_representation,
                                                 p_features=avg_lrp_scores_normalized,
                                                 activation=activation,
                                                 threshold=learn_p_weak_classifier_threshold)
            learn_committee_lrp.fit(x_train, y_train, learn_np_seed, learn_random_state)

        # Validation
        print("Validating Learn++...")
        learn_predictions.append(learn_committee.predict(x_test, label_df))
        learn_predictions_failure = []
        for learn_failure_test in x_test_failures:
            learn_predictions_failure.append(learn_committee.predict(learn_failure_test, label_df))
        learn_predictions_failure_set.append(learn_predictions_failure)
        if algorithms_to_execute["LRP"]:
            learn_predictions_lrp.append(learn_committee_lrp.predict(x_test, label_df))
            learn_predictions_failure_lrp = []
            for learn_failure_test_lrp in x_test_failures:
                learn_predictions_failure_lrp.append(learn_committee_lrp.predict(learn_failure_test_lrp, label_df))
            learn_predictions_failure_lrp_set.append(learn_predictions_failure_lrp)

# --- DropIn --- #
if algorithms_to_execute["DropIn"]:
    dropin_predictions_failure_set = []
    dropin_predictions = []
    for k in range(p_dropin_standard.__len__()):
        # Standard
        print("Training Standard DropIn...")
        dropin_network = DropInNetwork(hidden_layer_sizes=dropin_hidden_layer_sizes,
                                       learning_rate_init=dropin_learning_rate_init,
                                       p_dropin=p_dropin_standard[k],
                                       random_state=dropin_random_state,
                                       activation=activation)
        dropin_network.fit_dropin(x_train, y_train, dropin_np_seed, dropin_epochs)

        print("Validating Standard DropIn...")
        dropin_predictions.append(dropin_network.predict(x_test))
        dropin_predictions_failure = []
        for dropin_failure_test in x_test_failures:
            dropin_predictions_failure.append(dropin_network.predict(dropin_failure_test))
        dropin_predictions_failure_set.append(dropin_predictions_failure)

    if algorithms_to_execute["LRP"]:
        # LRP
        print("Training LRP DropIn (scaled)...")
        dropin_network_lrp = DropInNetwork(hidden_layer_sizes=dropin_hidden_layer_sizes,
                                           learning_rate_init=dropin_learning_rate_init,
                                           p_dropin=avg_lrp_scores_scaled_inverted,
                                           random_state=dropin_random_state,
                                           activation=activation)
        dropin_network_lrp.fit_dropin(x_train, y_train, dropin_np_seed, dropin_epochs)

        # LRP - Range
        print("Training LRP DropIn (range)...")
        dropin_network_lrp_r = DropInNetwork(hidden_layer_sizes=dropin_hidden_layer_sizes,
                                             learning_rate_init=dropin_learning_rate_init,
                                             p_dropin=avg_lrp_scores_range_inverted,
                                             random_state=dropin_random_state,
                                             activation=activation)
        dropin_network_lrp_r.fit_dropin(x_train, y_train, dropin_np_seed, dropin_epochs)

    # Validation
    print("Validating LRP DropIn...")
    if algorithms_to_execute["LRP"]:
        dropin_predictions_lrp = dropin_network_lrp.predict(x_test)
        dropin_predictions_failure_lrp = []
        for dropin_failure_test_lrp in x_test_failures:
            dropin_predictions_failure_lrp.append(dropin_network_lrp.predict(dropin_failure_test_lrp))
        dropin_predictions_lrp_r = dropin_network_lrp_r.predict(x_test)
        dropin_predictions_failure_lrp_r = []
        for dropin_failure_test_lrp_r in x_test_failures:
            dropin_predictions_failure_lrp_r.append(dropin_network_lrp.predict(dropin_failure_test_lrp_r))

# --- Selective Retraining --- #
if algorithms_to_execute["SelectiveRetraining"]:
    sr_original_predictions = []
    sr_original_predictions_failure_set = []
    sr_predictions = []
    sr_predictions_failure_set = []
    sr_predictions_failure_lrp_set = []
    for j in range(sr_weight_threshold.__len__()):
        print("Training Selective Retraining Committee", j, "...")
        selective_committee = SelectiveRetrainingCommittee(learning_rate_init=sr_learning_rate_init,
                                                           hidden_layer_sizes=sr_hidden_layer_sizes,
                                                           random_state=sr_random_state,
                                                           activation=activation,
                                                           labels=labels)
        selective_committee.fit(x_train, y_train, sr_weight_threshold[j])

        # Validation
        print("Validating Selective Retraining...")
        sr_original_predictions.append(selective_committee.predict_without_retraining(x_test))
        sr_original_predictions_failure = []
        for sr_original_failure_test in x_test_failures:
            sr_original_predictions_failure.append(selective_committee.predict_without_retraining(sr_original_failure_test))
        sr_original_predictions_failure_set.append(sr_original_predictions_failure)
        sr_predictions = selective_committee.predict(x_test, label_df)
        sr_predictions_failure = []
        for sr_failure_test in x_test_failures:
            sr_predictions_failure.append(selective_committee.predict(sr_failure_test, label_df))
        sr_predictions_failure_set.append(sr_predictions_failure)
        if algorithms_to_execute["LRP"]:
            sr_predictions_lrp = selective_committee.predict(x_test, label_df, avg_lrp_scores_normalized_inverted)
            sr_predictions_failure_lrp = []
            for sr_failure_test_lrp in x_test_failures:
                sr_predictions_failure_lrp.append(
                    selective_committee.predict(sr_failure_test_lrp, label_df, avg_lrp_scores_normalized_inverted))
            sr_predictions_failure_lrp_set.append(sr_predictions_failure_lrp)

# --- PRINT RESULTS --- #
print("Data Set: ", data_set)
if algorithms_to_execute["LRP"]:
    print("Number of tuples taken into consideration:")
    print(lrp_nn.LRP_scores_regarded)
    print("Average LRP Scores per Feature:")
    print(avg_lrp_scores)
    print("Normalized - to be used for Learn++:")
    print(avg_lrp_scores_normalized)
    print("Inverted normalized - to be used for Selective Retraining:")
    print(avg_lrp_scores_normalized_inverted)
    print("Scaled to Dropout probabilities:")
    print(avg_lrp_scores_scaled)
    print("Inverted to Dropin probabilities - to be used for DropIn:")
    print(avg_lrp_scores_scaled_inverted)
    print("Scaled by range to Dropout prob.:")
    print(avg_lrp_scores_range)
    print("Inverted range prob.:")
    print(avg_lrp_scores_range_inverted)

    print("Accuracy Score - LRP Network: ")
    print("           w/o Sensor Failure: ", accuracy_score(lrp_predictions, y_test))
    for i in range(lrp_predictions_failure.__len__()):
        print("           w/  Sensor Failure (", failure_percentages[i], "): ", accuracy_score(lrp_predictions_failure[i], y_test))

if algorithms_to_execute["Learn++"]:
    print("Accuracy Score - Learn++:")
    for pof in range(learn_percentage_of_features.__len__()):
        print("Percentage of features: ", learn_percentage_of_features[pof])
        print("w/o LRP  & w/o Sensor Failure: ", accuracy_score(learn_predictions[pof], y_test))
        for i in range(learn_predictions_failure_set[pof].__len__()):
            print("w/o LRP  & w/  Sensor Failure (", failure_percentages[i], "): ", accuracy_score(learn_predictions_failure_set[pof][i], y_test))
        if algorithms_to_execute["LRP"]:
            print("w/ LRP   & w/o Sensor Failure: ", accuracy_score(learn_predictions_lrp[pof], y_test))
            for i in range(learn_predictions_failure_lrp_set[pof].__len__()):
                print("w/ LRP   & w/  Sensor Failure(", failure_percentages[i], "): ", accuracy_score(learn_predictions_failure_lrp[pof][i], y_test))

if algorithms_to_execute["DropIn"]:
    print("Accuracy Score - DropIn:")
    for k in range(p_dropin_standard.__len__()):
        print("DropIn Prob.:", p_dropin_standard[k])
        print("w/o LRP  & w/o Sensor Failure: ", accuracy_score(dropin_predictions[k], y_test))
        for i in range(dropin_predictions_failure_set[k].__len__()):
            print("w/o LRP  & w/  Sensor Failure (", failure_percentages[i], "): ", accuracy_score(dropin_predictions_failure_set[k][i], y_test))
    if algorithms_to_execute["LRP"]:
        print("w/  LRP  & w/o Sensor Failure: ", accuracy_score(dropin_predictions_lrp[k], y_test))
        for i in range(dropin_predictions_failure_lrp.__len__()):
            print("w/  LRP  & w/  Sensor Failure (", failure_percentages[i], "): ", accuracy_score(dropin_predictions_failure_lrp[i], y_test))
        print("w/  LRPr & w/o Sensor Failure: ", accuracy_score(dropin_predictions_lrp_r, y_test))
        for i in range(dropin_predictions_failure_lrp_r.__len__()):
            print("w/  LRPr & w/  Sensor Failure (", failure_percentages[i], "): ", accuracy_score(dropin_predictions_failure_lrp_r[i], y_test))

if algorithms_to_execute["SelectiveRetraining"]:
    for j in range(sr_weight_threshold.__len__()):
        print("Accuracy Score - Selective Retraining (Weight Threshold: ", sr_weight_threshold[j],"):")
        print("Accuracy Score - Without Retraining: ")
        print("           w/o Sensor Failure: ", accuracy_score(sr_original_predictions[j], y_test))
        for i in range(sr_original_predictions_failure[j].__len__()):
            print("           w/  Sensor Failure", failure_percentages[i],": ", accuracy_score(sr_original_predictions_failure[j][i], y_test))
        print("Accuracy Score - Selective Retraining: ")
        print("w/o LRP  & w/o Sensor Failure: ", accuracy_score(sr_predictions[j], y_test))
        for i in range(sr_predictions_failure[j].__len__()):
            print("w/o LRP  & w/  Sensor Failure (", failure_percentages[i], ": ", accuracy_score(sr_predictions_failure[j][i], y_test))
        if algorithms_to_execute["LRP"]:
            print("w/  LRP  & w/o Sensor Failure: ", accuracy_score(sr_predictions_lrp[j], y_test))
            for i in range(sr_predictions_failure_lrp[j].__len__()):
                print("w/  LRP  & w/  Sensor Failure (", failure_percentages[i], "): ", accuracy_score(sr_predictions_failure_lrp[j][i], y_test))
