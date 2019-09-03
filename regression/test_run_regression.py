import regression.data_lib_regression as data_lib
import imputation
from regression.LRP_regression import LRPNetworkRegression as LRPNetwork
from regression.LearnPlus_regression import LearnCommitteeRegression as LearnCommittee
from regression.DropIn_regression import DropInNetworkRegression as DropInNetwork
from regression.SelectiveRetraining_regression import SelectiveRetrainingCommitteeRegression \
    as SelectiveRetrainingCommittee
from sklearn import model_selection
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

# --- PARAMETERS --- #
# General
algorithms_to_execute = {"LRP":     True,
                         "Learn++": True,
                         "DropIn":  True,
                         "SelectiveRetraining": True,
                         "Imputation": True,
                         "Failure_Known": True}
data_set = "sklearn"
data_set_params = {"n_samples":      5000,
                   "n_features":     15,
                   "n_informative":  15,
                   "n_targets":      1,
                   "bias":           0.0,
                   "effective_rank": None,
                   "tail_strength":  0.5,
                   "noise":          0.3,
                   "data_shuffle":   True,
                   "coef":           False,
                   "random_state":   7}

ms_random_state = 9
ms_test_size = 0.1
failure_simulation_np_seed = 7
failure_percentages = [0.10, 0.25, 0.5, 0.75, 0.90]
random_failure = False
multi_sensor_failure = True
n_nearest_neighbors = 3

# LRP
LRP_hidden_layer_sizes = [45, 45, 45]
LRP_learning_rate_init = 0.1
LRP_random_states = [7]
LRP_seed = 9
LRP_test_size = 0.1
LRP_alpha = 2
LRP_r2_threshold = 0.3
LRP_dropout_threshold_max = 0.4
LRP_dropout_threshold_min = 0.1

# Imputation
imputation_hidden_layer_sizes = [45, 45, 45]
imputation_learning_rate_init = 0.1
imputation_random_state = 3

# Learn++
learn_hidden_layer_sizes = [20, 20, 20]
learn_learning_rate_init = 0.1
learn_random_state = 5
learn_np_seed = 7
learn_no_of_weak_regressors = [50, 70]
learn_percentage_of_features = [0.15, 0.2, 0.4, 0.6]
learn_missing_data_representation = None
learn_p_features_standard = None
learn_p_weak_regressor_threshold = 0.8

# DropIn
dropin_hidden_layer_sizes = [30, 30, 30]
dropin_learning_rate_init = 0.1
dropin_random_state = 9
dropin_np_seed = 12
p_dropin_standard = [0.2, 0.8]
dropin_epochs = 6
dropin_seq_length = 5

# Selective Retraining
sr_hidden_layer_sizes = [45, 45, 45]
sr_learning_rate_init = 0.1
sr_random_state = 7
sr_weight_threshold = [0.1, 0.25, 0.3, 0.5, 0.75]

# --- Data Preparation --- #
X, Y, activation, probabilities = \
    data_lib.get_data_set(data_set,
                          n_samples=data_set_params["n_samples"],
                          n_features=data_set_params["n_features"],
                          n_informative=data_set_params["n_informative"],
                          n_targets=data_set_params["n_targets"],
                          bias=data_set_params["bias"],
                          effective_rank=data_set_params["effective_rank"],
                          tail_strength=data_set_params["tail_strength"],
                          noise=data_set_params["noise"],
                          data_shuffle=data_set_params["data_shuffle"],
                          coef=data_set_params["coef"],
                          random_state=data_set_params["random_state"])

x_train, x_test, y_train, y_test = \
    model_selection.train_test_split(X, Y, test_size=ms_test_size, random_state=ms_random_state)

x_test_failures = [data_lib.get_sensor_failure_test_set(x_test,
                                                        np_seed=failure_simulation_np_seed,
                                                        random=random_failure,
                                                        multi_sensor_failure=multi_sensor_failure,
                                                        failure_percentage=p)
                   for p in failure_percentages]
x_known_test_failures = None
if algorithms_to_execute["Failure_Known"]:
    x_known_test_failures = data_lib.get_sensor_failure_test_set(x_test,
                                                                 np_seed=failure_simulation_np_seed,
                                                                 probability_known=True,
                                                                 probabilities=probabilities)

# --- Imputation --- #
knn_imputation_predictions_failure = []
median_imputation_predictions_failure = []
mean_imputation_predictions_failure = []
imputation_predictions = []

if algorithms_to_execute["Imputation"]:
    print("Create Imputations...")
    x_test_knn_imputations = [imputation.knn_imputation(x, n_nearest_neighbors) for x in x_test_failures]
    x_test_median_imputations = [imputation.mean_imputation(x) for x in x_test_failures]
    x_test_mean_imputations = [imputation.median_imputation(x) for x in x_test_failures]

    x_test_median_known = None
    x_test_mean_known = None
    x_test_knn_known = None
    if algorithms_to_execute["Failure_Known"]:
        x_test_knn_known = imputation.knn_imputation(x_known_test_failures, n_nearest_neighbors)
        x_test_median_known = imputation.median_imputation(x_known_test_failures)
        x_test_mean_known = imputation.mean_imputation(x_known_test_failures)

    print("Training Imputation Network...")
    imputation_nn = MLPRegressor(hidden_layer_sizes=imputation_hidden_layer_sizes,
                                 learning_rate_init=imputation_learning_rate_init,
                                 activation=activation,
                                 random_state=imputation_random_state)
    imputation_nn.fit(x_train, y_train)

    print("Validating Imputation NN...")
    imputation_predictions = imputation_nn.predict(x_test)
    for knn_failure_test in x_test_knn_imputations:
        knn_imputation_predictions_failure.append(imputation_nn.predict(knn_failure_test))
    for mean_failure_test in x_test_mean_imputations:
        mean_imputation_predictions_failure.append(imputation_nn.predict(mean_failure_test))
    for median_failure_test in x_test_median_imputations:
        median_imputation_predictions_failure.append(imputation_nn.predict(median_failure_test))
    if algorithms_to_execute["Failure_Known"]:
        knn_pred_failure_known = imputation_nn.predict(x_test_knn_known)
        median_pred_failure_known = imputation_nn.predict(x_test_median_known)
        mean_pred_failure_known = imputation_nn.predict(x_test_mean_known)

# --- LRP Score Calculation --- #
avg_lrp_scores = None
avg_lrp_scores_range = None
avg_lrp_scores_range_inverted = None
avg_lrp_scores_normalized = None
avg_lrp_scores_normalized_inverted = None
avg_lrp_scores_scaled = None
avg_lrp_scores_scaled_inverted = None
lrp_nn = None
lrp_predictions = None
lrp_predictions_failure = None

if algorithms_to_execute["LRP"]:
    print("Training LRP Network...")
    lrp_nn = LRPNetwork(hidden_layer_sizes=LRP_hidden_layer_sizes,
                        learning_rate_init=LRP_learning_rate_init,
                        no_of_in_nodes=len(X[0]),
                        activation=activation)
    print("Calculating LRP Scores...")
    avg_lrp_scores, mlp_network = lrp_nn.avg_lrp_score_per_feature(features=X,
                                                                   target_values=Y,
                                                                   test_size=LRP_test_size,
                                                                   seed=LRP_seed,
                                                                   random_states=LRP_random_states,
                                                                   alpha=LRP_alpha,
                                                                   threshold=LRP_r2_threshold)
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
learn_predictions = []
learn_predictions_lrp = []
learn_predictions_lrp_inverted = []
learn_predictions_failure_set = []
learn_predictions_failure_lrp_set = []
learn_predictions_failure_lrp_inverted_set = []
learn_predictions_known = []
learn_predictions_known_lrp = []
learn_predictions_known_lrp_inverted = []
learn_predictions_known_standard = []

if algorithms_to_execute["Learn++"]:
    # standard
    print("Training Standard Learn++ Committee...")
    for w in learn_no_of_weak_regressors:
        wlearn_predictions = []
        wlearn_predictions_lrp = []
        wlearn_predictions_lrp_inverted = []
        wlearn_predictions_failure_set = []
        wlearn_predictions_failure_lrp_set = []
        wlearn_predictions_failure_lrp_inverted_set = []
        wlearn_predictions_known = []
        wlearn_predictions_known_lrp = []
        wlearn_predictions_known_lrp_inverted = []
        wlearn_predictions_known_standard = []
        for pof in learn_percentage_of_features:
            learn_committee = LearnCommittee(no_of_weak_regressors=w,
                                             percentage_of_features=pof,
                                             no_of_features=len(X[0]),
                                             hidden_layer_sizes=learn_hidden_layer_sizes,
                                             learning_rate_init=learn_learning_rate_init,
                                             missing_data_representation=learn_missing_data_representation,
                                             p_features=learn_p_features_standard,
                                             activation=activation,
                                             threshold=learn_p_weak_regressor_threshold)
            learn_committee.fit(x_train, y_train, learn_np_seed, learn_random_state)

            learn_committee_lrp = None
            learn_committee_lrp_inverted = None
            if algorithms_to_execute["LRP"]:
                # LRP
                print("Training LRP Learn++ Committee...")
                learn_committee_lrp = LearnCommittee(no_of_weak_regressors=w,
                                                     percentage_of_features=pof,
                                                     no_of_features=len(X[0]),
                                                     hidden_layer_sizes=learn_hidden_layer_sizes,
                                                     learning_rate_init=learn_learning_rate_init,
                                                     missing_data_representation=learn_missing_data_representation,
                                                     p_features=avg_lrp_scores_normalized,
                                                     activation=activation,
                                                     threshold=learn_p_weak_regressor_threshold)
                learn_committee_lrp.fit(x_train, y_train, learn_np_seed, learn_random_state)

                learn_committee_lrp_inverted = \
                    LearnCommittee(no_of_weak_regressors=w,
                                   percentage_of_features=pof,
                                   no_of_features=len(X[0]),
                                   hidden_layer_sizes=learn_hidden_layer_sizes,
                                   learning_rate_init=learn_learning_rate_init,
                                   missing_data_representation=learn_missing_data_representation,
                                   p_features=avg_lrp_scores_normalized_inverted,
                                   activation=activation,
                                   threshold=learn_p_weak_regressor_threshold)
                learn_committee_lrp_inverted.fit(x_train, y_train, learn_np_seed, learn_random_state)

            learn_committee_known = None
            if algorithms_to_execute["Failure_Known"]:
                # Failure known
                print("Training Known Failure Learn++ Committee...")
                inverted_probabilities = [sum(probabilities) - x for x in probabilities]
                learn_committee_known = LearnCommittee(no_of_weak_regressors=w,
                                                       percentage_of_features=pof,
                                                       no_of_features=len(X[0]),
                                                       hidden_layer_sizes=learn_hidden_layer_sizes,
                                                       learning_rate_init=learn_learning_rate_init,
                                                       missing_data_representation=learn_missing_data_representation,
                                                       p_features=[x / sum(inverted_probabilities)
                                                                   for x in inverted_probabilities],
                                                       activation=activation,
                                                       threshold=learn_p_weak_regressor_threshold)
                learn_committee_known.fit(x_train, y_train, learn_np_seed, learn_random_state)

            # Validation
            print("Validating Learn++...")
            wlearn_predictions.append(learn_committee.predict(x_test))
            wlearn_predictions_failure = []
            for learn_failure_test in x_test_failures:
                wlearn_predictions_failure.append(learn_committee.predict(learn_failure_test))
            wlearn_predictions_failure_set.append(wlearn_predictions_failure)
            if algorithms_to_execute["LRP"]:
                wlearn_predictions_lrp.append(learn_committee_lrp.predict(x_test))
                wlearn_predictions_failure_lrp = []
                for learn_failure_test_lrp in x_test_failures:
                    wlearn_predictions_failure_lrp.append(learn_committee_lrp.predict(learn_failure_test_lrp))
                wlearn_predictions_failure_lrp_set.append(wlearn_predictions_failure_lrp)

                wlearn_predictions_lrp_inverted.append(learn_committee_lrp_inverted.predict(x_test))
                wlearn_predictions_failure_lrp_inverted = []
                for learn_failure_test_lrp in x_test_failures:
                    wlearn_predictions_failure_lrp_inverted.\
                        append(learn_committee_lrp_inverted.predict(learn_failure_test_lrp))
                wlearn_predictions_failure_lrp_inverted_set.append(wlearn_predictions_failure_lrp_inverted)
            if algorithms_to_execute["Failure_Known"]:
                wlearn_predictions_known.append(learn_committee_known.predict(x_known_test_failures))
                wlearn_predictions_known_lrp.append(learn_committee_lrp.predict(x_known_test_failures))
                wlearn_predictions_known_lrp_inverted.append(learn_committee_lrp_inverted.predict(x_known_test_failures))
                wlearn_predictions_known_standard.append(learn_committee.predict(x_known_test_failures))
        learn_predictions.append(wlearn_predictions)
        learn_predictions_lrp.append(wlearn_predictions_lrp)
        learn_predictions_lrp_inverted.append(wlearn_predictions_lrp_inverted)
        learn_predictions_failure_set.append(wlearn_predictions_failure_set)
        learn_predictions_failure_lrp_set.append(wlearn_predictions_failure_lrp_set)
        learn_predictions_failure_lrp_inverted_set.append(wlearn_predictions_failure_lrp_inverted_set)
        learn_predictions_known.append(wlearn_predictions_known)
        learn_predictions_known_lrp.append(wlearn_predictions_known_lrp)
        learn_predictions_known_lrp_inverted.append(wlearn_predictions_known_lrp_inverted)
        learn_predictions_known_standard.append(wlearn_predictions_known_standard)

# --- DropIn --- #
dropin_predictions = None
dropin_predictions_lrp = None
dropin_predictions_lrp_r = None
dropin_predictions_failure_set = None
dropin_predictions_failure_lrp = None
dropin_predictions_failure_lrp_r = None
dropin_predictions_failure_known = None
dropin_predictions_failure_known_lrp = None
dropin_predictions_failure_known_lrp_r = None
dropin_predictions_failure_known_standard = None
dropin_network = None
dropin_network_lrp = None
dropin_network_lrp_r = None

if algorithms_to_execute["DropIn"]:
    dropin_predictions = []
    dropin_predictions_failure_set = []
    for k in range(p_dropin_standard.__len__()):
        # Standard
        print("Training Standard DropIn...")
        dropin_network = DropInNetwork(hidden_layer_sizes=dropin_hidden_layer_sizes,
                                       learning_rate_init=dropin_learning_rate_init,
                                       p_dropin=p_dropin_standard[k],
                                       random_state=dropin_random_state,
                                       activation=activation)
        dropin_network.fit_dropin(x_train, y_train, dropin_np_seed, dropin_epochs, sequence_length=dropin_seq_length)

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
        dropin_network_lrp.fit_dropin(x_train, y_train, dropin_np_seed, dropin_epochs, sequence_length=dropin_seq_length)

        # LRP - Range
        print("Training LRP DropIn (range)...")
        dropin_network_lrp_r = DropInNetwork(hidden_layer_sizes=dropin_hidden_layer_sizes,
                                             learning_rate_init=dropin_learning_rate_init,
                                             p_dropin=avg_lrp_scores_range_inverted,
                                             random_state=dropin_random_state,
                                             activation=activation)
        dropin_network_lrp_r.fit_dropin(x_train, y_train, dropin_np_seed, dropin_epochs, sequence_length=dropin_seq_length)

        # Validation
        print("Validating LRP DropIn...")
        dropin_predictions_lrp = dropin_network_lrp.predict(x_test)
        dropin_predictions_failure_lrp = []
        for dropin_failure_test_lrp in x_test_failures:
            dropin_predictions_failure_lrp.append(dropin_network_lrp.predict(dropin_failure_test_lrp))
        dropin_predictions_lrp_r = dropin_network_lrp_r.predict(x_test)
        dropin_predictions_failure_lrp_r = []
        for dropin_failure_test_lrp_r in x_test_failures:
            dropin_predictions_failure_lrp_r.append(dropin_network_lrp.predict(dropin_failure_test_lrp_r))

    if algorithms_to_execute["Failure_Known"]:
        # Failure known
        print("Training Known Failure DropIn ...")
        probabilities_inverted = [1 - x for x in probabilities]
        dropin_network_failure_known = DropInNetwork(hidden_layer_sizes=dropin_hidden_layer_sizes,
                                                     learning_rate_init=dropin_learning_rate_init,
                                                     p_dropin=probabilities_inverted,
                                                     random_state=dropin_random_state,
                                                     activation=activation)
        dropin_network_failure_known.fit_dropin(x_train, y_train, dropin_np_seed, dropin_epochs, sequence_length=dropin_seq_length)
        dropin_predictions_failure_known = dropin_network_failure_known.predict(x_known_test_failures)
        dropin_predictions_failure_known_lrp = dropin_network_lrp.predict(x_known_test_failures)
        dropin_predictions_failure_known_lrp_r = dropin_network_lrp_r.predict(x_known_test_failures)
        dropin_predictions_failure_known_standard = dropin_network.predict(x_known_test_failures)

# --- Selective Retraining --- #
sr_original_predictions = []
sr_original_predictions_failure_set = []
sr_predictions = []
sr_predictions_lrp = []
sr_predictions_failure_set = []
sr_predictions_failure_lrp_set = []

if algorithms_to_execute["SelectiveRetraining"]:
    for j in range(sr_weight_threshold.__len__()):
        print("Training Selective Retraining Committee", j, "...")
        selective_committee = SelectiveRetrainingCommittee(learning_rate_init=sr_learning_rate_init,
                                                           hidden_layer_sizes=sr_hidden_layer_sizes,
                                                           random_state=sr_random_state,
                                                           activation=activation)
        selective_committee.fit(x_train, y_train, sr_weight_threshold[j])

        # Validation
        print("Validating Selective Retraining...")
        sr_original_predictions.append(selective_committee.predict_without_retraining(x_test))
        sr_original_predictions_failure = []
        for sr_original_failure_test in x_test_failures:
            sr_original_predictions_failure\
                .append(selective_committee.predict_without_retraining(sr_original_failure_test))
        sr_original_predictions_failure_set.append(sr_original_predictions_failure)
        sr_predictions.append(selective_committee.predict(x_test))
        sr_predictions_failure = []
        for sr_failure_test in x_test_failures:
            sr_predictions_failure.append(selective_committee.predict(sr_failure_test))
        sr_predictions_failure_set.append(sr_predictions_failure)
        if algorithms_to_execute["LRP"]:
            sr_predictions_lrp.append(selective_committee.predict(x_test, avg_lrp_scores_normalized_inverted))
            sr_predictions_failure_lrp = []
            for sr_failure_test_lrp in x_test_failures:
                sr_predictions_failure_lrp.append(
                    selective_committee.predict(sr_failure_test_lrp, avg_lrp_scores_normalized_inverted))
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

    print("")
    print("R2 Score - LRP Network: ")
    print("           w/o Sensor Failure:         ", r2_score(lrp_predictions, y_test))
    for i in range(lrp_predictions_failure.__len__()):
        print("           w/  Sensor Failure (", failure_percentages[i], "): ",
              r2_score(lrp_predictions_failure[i], y_test))
    print("MSE Score - LRP Network: ")
    print("           w/o Sensor Failure:         ", mean_squared_error(lrp_predictions, y_test))
    for i in range(lrp_predictions_failure.__len__()):
        print("           w/  Sensor Failure (", failure_percentages[i], "): ",
              mean_squared_error(lrp_predictions_failure[i], y_test))

if algorithms_to_execute["Imputation"]:
    print("")
    print("MSE - Imputation:")
    print("          w/o Sensor Failure:            ", mean_squared_error(imputation_predictions, y_test))
    for i in range(median_imputation_predictions_failure.__len__()):
        print("          w/  Median Imputation (", failure_percentages[i], "): ",
              mean_squared_error(median_imputation_predictions_failure[i], y_test))
    for i in range(mean_imputation_predictions_failure.__len__()):
        print("          w/  Mean Imputation   (", failure_percentages[i], "): ",
              mean_squared_error(mean_imputation_predictions_failure[i], y_test))
    for i in range(knn_imputation_predictions_failure.__len__()):
        print("          w/  kNN Imputation    (", failure_percentages[i], "): ",
              mean_squared_error(knn_imputation_predictions_failure[i], y_test))
    if algorithms_to_execute["Failure_Known"]:
        print("          w/  Median Imputation & w/ Failure Known: ", mean_squared_error(median_pred_failure_known, y_test))
        print("          w/  Mean Imputation & w/ Failure Known:   ", mean_squared_error(mean_pred_failure_known, y_test))
        print("          w/  kNN Imputation & w/ Failure Known:    ", mean_squared_error(knn_pred_failure_known, y_test))

if algorithms_to_execute["Learn++"]:
    print("")
    print("MSE - Learn++:")
    for w in range(learn_no_of_weak_regressors.__len__()):
        print("Number of weak regressors: ", learn_no_of_weak_regressors[w])
        for pof in range(learn_percentage_of_features.__len__()):
            print("Percentage of features: ", learn_percentage_of_features[pof])
            print("w/o LRP  & w/o Sensor Failure:           ",
                  mean_squared_error(learn_predictions[w][pof], y_test))
            for i in range(learn_predictions_failure_set[w][pof].__len__()):
                print("w/o LRP  & w/  Sensor Failure (", failure_percentages[i], "):   ",
                      mean_squared_error(learn_predictions_failure_set[w][pof][i], y_test))
            if algorithms_to_execute["LRP"]:
                print("w/ LRP   & w/o Sensor Failure:           ",
                      mean_squared_error(learn_predictions_lrp[w][pof], y_test))
                for i in range(learn_predictions_failure_lrp_set[w][pof].__len__()):
                    print("w/ LRP   & w/  Sensor Failure(", failure_percentages[i], "):    ",
                          mean_squared_error(learn_predictions_failure_lrp_set[w][pof][i], y_test))
                print("w/ LRPinv   & w/o Sensor Failure:        ",
                      mean_squared_error(learn_predictions_lrp[w][pof], y_test))
                for i in range(learn_predictions_failure_lrp_inverted_set[w][pof].__len__()):
                    print("w/ LRPinv   & w/  Sensor Failure(", failure_percentages[i], "): ",
                          mean_squared_error(learn_predictions_failure_lrp_inverted_set[w][pof][i], y_test))
            if algorithms_to_execute["Failure_Known"]:
                print("w/ Failure Known:                        ",
                      mean_squared_error(learn_predictions_known[w][pof], y_test))
                print("w/ Failure Known & w/ LRP:               ",
                      mean_squared_error(learn_predictions_known_lrp[w][pof], y_test))
                print("w/ Failure Known & w/ LRPinv:            ",
                      mean_squared_error(learn_predictions_known_lrp_inverted[w][pof], y_test))
                print("w/ Failure Known & w/ Standard Training: ",
                      mean_squared_error(learn_predictions_known_standard[w][pof], y_test))

if algorithms_to_execute["DropIn"]:
    print("")
    print("MSE - DropIn:")
    for k in range(p_dropin_standard.__len__()):
        print("DropIn Prob.:", p_dropin_standard[k])
        print("w/o LRP  & w/o Sensor Failure:         ",mean_squared_error(dropin_predictions[k], y_test))
        for i in range(dropin_predictions_failure_set[k].__len__()):
            print("w/o LRP  & w/  Sensor Failure (", failure_percentages[i], "): ",
                  mean_squared_error(dropin_predictions_failure_set[k][i], y_test))
    if algorithms_to_execute["LRP"]:
        print("w/  LRP  & w/o Sensor Failure:         ", mean_squared_error(dropin_predictions_lrp, y_test))
        for i in range(dropin_predictions_failure_lrp.__len__()):
            print("w/  LRP  & w/  Sensor Failure (", failure_percentages[i], "): ",
                  mean_squared_error(dropin_predictions_failure_lrp[i], y_test))
        print("w/  LRPr & w/o Sensor Failure:         ", mean_squared_error(dropin_predictions_lrp_r, y_test))
        for i in range(dropin_predictions_failure_lrp_r.__len__()):
            print("w/  LRPr & w/  Sensor Failure (", failure_percentages[i], "): ",
                  mean_squared_error(dropin_predictions_failure_lrp_r[i], y_test))
    if algorithms_to_execute["Failure_Known"]:
        print("w/ Failure Known:                       ",
              mean_squared_error(dropin_predictions_failure_known, y_test))
        print("w/ Failure Known & w/ LRP:              ",
              mean_squared_error(dropin_predictions_failure_known_lrp, y_test))
        print("w/ Failure Known & w/ LRPr:             ",
              mean_squared_error(dropin_predictions_failure_known_lrp_r, y_test))
        print("w/ Failure Known & w/ Standard Training:",
              mean_squared_error(dropin_predictions_failure_known_standard, y_test))

if algorithms_to_execute["SelectiveRetraining"]:
    print("")
    for j in range(sr_weight_threshold.__len__()):
        print("MSE - Selective Retraining (Weight Threshold: ", sr_weight_threshold[j], "):")
        print("MSE - Without Retraining: ")
        print("           w/o Sensor Failure:         ", mean_squared_error(sr_original_predictions[j], y_test))
        for i in range(sr_original_predictions_failure_set[j].__len__()):
            print("           w/  Sensor Failure (", failure_percentages[i], "): ",
                  mean_squared_error(sr_original_predictions_failure_set[j][i], y_test))
        print("MSE - Selective Retraining: ")
        print("w/o LRP  & w/o Sensor Failure:         ", mean_squared_error(sr_predictions[j], y_test))
        for i in range(sr_predictions_failure_set[j].__len__()):
            print("w/o LRP  & w/  Sensor Failure (", failure_percentages[i], "): ",
                  mean_squared_error(sr_predictions_failure_set[j][i], y_test))
        if algorithms_to_execute["LRP"]:
            print("w/  LRP  & w/o Sensor Failure:         ", mean_squared_error(sr_predictions_lrp[j], y_test))
            for i in range(sr_predictions_failure_lrp_set[j].__len__()):
                print("w/  LRP  & w/  Sensor Failure (", failure_percentages[i], "): ",
                      mean_squared_error(sr_predictions_failure_lrp_set[j][i], y_test))
