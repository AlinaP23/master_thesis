import regression.data_lib_regression as data_lib
from regression.LRP_regression import LRPNetworkRegression as LRPNetwork
from regression.LearnPlus_regression import LearnCommitteeRegression as LearnCommittee
from regression.DropIn_regression import DropInNetworkRegression as DropInNetwork
from regression.SelectiveRetraining_regression import SelectiveRetrainingCommitteeRegression \
    as SelectiveRetrainingCommittee
from sklearn import model_selection
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# --- PARAMETERS --- #
# General
algorithms_to_execute = {"LRP":     False,
                         "Learn++": False,
                         "DropIn":  False,
                         "SelectiveRetraining": True}
data_set = "sklearn"
data_set_params = {"n_samples":      50000,
                   "n_features":     15,
                   "n_informative":  9,
                   "n_targets":      1,
                   "bias":           0.0,
                   "effective_rank": None,
                   "tail_strength":  0.5,
                   "noise":          0.0,
                   "data_shuffle":   True,
                   "coef":           False,
                   "random_state":   7}
ms_random_state = 9
ms_test_size = 0.1
failure_simulation_np_seed = 7
failure_percentage = 0.2
random_failure = False
multi_sensor_failure = True

X, Y, activation = data_lib.get_data_set(data_set,
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
x_test_failure = data_lib.get_sensor_failure_test_set(x_test,
                                                      np_seed=failure_simulation_np_seed,
                                                      random=random_failure,
                                                      multi_sensor_failure=multi_sensor_failure,
                                                      failure_percentage=failure_percentage)
# LRP
LRP_hidden_layer_sizes = [20, 20, 20]
LRP_learning_rate_init = 0.1
LRP_random_states = [7]
LRP_seed = 9
LRP_test_size = 0.1
LRP_alpha = 2
LRP_r2_threshold = 0.3
LRP_dropout_threshold_max = 0.4
LRP_dropout_threshold_min = 0.1

# Learn++
learn_hidden_layer_sizes = [5, 5, 5]
learn_learning_rate_init = 0.1
learn_random_state = 9
learn_np_seed = 9
learn_no_of_weak_regressors = 5
learn_percentage_of_features = 0.75
learn_missing_data_representation = None
learn_p_features_standard = None
learn_p_weak_regressor_threshold = 0.3

# DropIn
dropin_hidden_layer_sizes = [80, 80, 80]
dropin_learning_rate_init = 0.1
dropin_random_state = 7
dropin_np_seed = 12
p_dropin_standard = 0

# Selective Retraining
sr_hidden_layer_sizes = [30, 30, 30]
sr_learning_rate_init = 0.1
sr_random_state = 9
sr_weight_threshold = 0.7

# --- LRP Score Calculation --- #
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
    lrp_predictions_failure = mlp_network.predict(x_test_failure)

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
    learn_committee = LearnCommittee(no_of_weak_regressors=learn_no_of_weak_regressors,
                                     percentage_of_features=learn_percentage_of_features,
                                     no_of_features=len(X[0]),
                                     hidden_layer_sizes=learn_hidden_layer_sizes,
                                     learning_rate_init=learn_learning_rate_init,
                                     missing_data_representation=learn_missing_data_representation,
                                     p_features=learn_p_features_standard,
                                     activation=activation,
                                     threshold=learn_p_weak_regressor_threshold)
    learn_committee.fit(x_train, y_train, learn_np_seed, learn_random_state)

    if algorithms_to_execute["LRP"]:
        # LRP
        print("Training LRP Learn++ Committee...")
        learn_committee_lrp = LearnCommittee(no_of_weak_regressors=learn_no_of_weak_regressors,
                                             percentage_of_features=learn_percentage_of_features,
                                             no_of_features=len(X[0]),
                                             hidden_layer_sizes=learn_hidden_layer_sizes,
                                             learning_rate_init=learn_learning_rate_init,
                                             missing_data_representation=learn_missing_data_representation,
                                             p_features=avg_lrp_scores_normalized,
                                             activation=activation,
                                             threshold=learn_p_weak_regressor_threshold)
        learn_committee_lrp.fit(x_train, y_train, learn_np_seed, learn_random_state)

    # Validation
    print("Validating Learn++...")
    learn_predictions = learn_committee.predict(x_test)
    learn_predictions_failure = learn_committee.predict(x_test_failure)
    if algorithms_to_execute["LRP"]:
        learn_predictions_lrp = learn_committee_lrp.predict(x_test)
        learn_predictions_failure_lrp = learn_committee_lrp.predict(x_test_failure)

# --- DropIn --- #
if algorithms_to_execute["DropIn"]:
    # Standard
    print("Training Standard DropIn...")
    dropin_network = DropInNetwork(hidden_layer_sizes=dropin_hidden_layer_sizes,
                                   learning_rate_init=dropin_learning_rate_init,
                                   p_dropin=p_dropin_standard,
                                   random_state=dropin_random_state,
                                   activation=activation)
    dropin_network.fit_dropin(x_train, y_train, dropin_np_seed)

    if algorithms_to_execute["LRP"]:
        # LRP
        print("Training LRP DropIn (scaled)...")
        dropin_network_lrp = DropInNetwork(hidden_layer_sizes=dropin_hidden_layer_sizes,
                                           learning_rate_init=dropin_learning_rate_init,
                                           p_dropin=avg_lrp_scores_scaled_inverted,
                                           random_state=dropin_random_state,
                                           activation=activation)
        dropin_network_lrp.fit_dropin(x_train, y_train, dropin_np_seed)

        # LRP - Range
        print("Training LRP DropIn (range)...")
        dropin_network_lrp_r = DropInNetwork(hidden_layer_sizes=dropin_hidden_layer_sizes,
                                             learning_rate_init=dropin_learning_rate_init,
                                             p_dropin=avg_lrp_scores_range_inverted,
                                             random_state=dropin_random_state,
                                             activation=activation)
        dropin_network_lrp_r.fit_dropin(x_train, y_train, dropin_np_seed)

    # Validation
    print("Validating DropIn...")
    dropin_predictions = dropin_network.predict(x_test)
    dropin_predictions_failure = dropin_network.predict(x_test_failure)
    if algorithms_to_execute["LRP"]:
        dropin_predictions_lrp = dropin_network_lrp.predict(x_test)
        dropin_predictions_failure_lrp = dropin_network_lrp.predict(x_test_failure)
        dropin_predictions_lrp_r = dropin_network_lrp_r.predict(x_test)
        dropin_predictions_failure_lrp_r = dropin_network_lrp_r.predict(x_test_failure)

# --- Selective Retraining --- #
if algorithms_to_execute["SelectiveRetraining"]:
    print("Training Selective Retraining Committee...")
    selective_committee = SelectiveRetrainingCommittee(learning_rate_init=sr_learning_rate_init,
                                                       hidden_layer_sizes=sr_hidden_layer_sizes,
                                                       random_state=sr_random_state,
                                                       activation=activation)
    selective_committee.fit(x_train, y_train, sr_weight_threshold)

    # Validation
    print("Validating Selective Retraining...")
    sr_original_predictions = selective_committee.predict_without_retraining(x_test)
    sr_original_predictions_failure = selective_committee.predict_without_retraining(x_test_failure)
    sr_predictions = selective_committee.predict(x_test)
    sr_predictions_failure = selective_committee.predict(x_test_failure)
    if algorithms_to_execute["LRP"]:
        sr_predictions_lrp = selective_committee.predict(x_test, avg_lrp_scores_normalized_inverted)
        sr_predictions_failure_lrp = \
            selective_committee.predict(x_test_failure, avg_lrp_scores_normalized_inverted)

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

    print("R2 Score - LRP Network: ")
    print("           w/o Sensor Failure: ", r2_score(lrp_predictions, y_test))
    print("           w/  Sensor Failure: ", r2_score(lrp_predictions_failure, y_test))
    print("MSE Score - LRP Network: ")
    print("           w/o Sensor Failure: ", mean_squared_error(lrp_predictions, y_test))
    print("           w/  Sensor Failure: ", mean_squared_error(lrp_predictions_failure, y_test))

if algorithms_to_execute["Learn++"]:
    print("R2 Score - Learn++:")
    print("w/o LRP  & w/o Sensor Failure: ", r2_score(learn_predictions, y_test))
    print("w/o LRP  & w/  Sensor Failure: ", r2_score(learn_predictions_failure, y_test))
    if algorithms_to_execute["LRP"]:
        print("w/ LRP   & w/o Sensor Failure: ", r2_score(learn_predictions_lrp, y_test))
        print("w/ LRP   & w/  Sensor Failure: ", r2_score(learn_predictions_failure_lrp, y_test))

    print("MSE Score - Learn++:")
    print("w/o LRP  & w/o Sensor Failure: ", mean_squared_error(learn_predictions, y_test))
    print("w/o LRP  & w/  Sensor Failure: ", mean_squared_error(learn_predictions_failure, y_test))
    if algorithms_to_execute["LRP"]:
        print("w/ LRP   & w/o Sensor Failure: ", mean_squared_error(learn_predictions_lrp, y_test))
        print("w/ LRP   & w/  Sensor Failure: ", mean_squared_error(learn_predictions_failure_lrp, y_test))

if algorithms_to_execute["DropIn"]:
    print("R2 Score - DropIn:")
    print("w/o LRP  & w/o Sensor Failure: ", r2_score(dropin_predictions, y_test))
    print("w/o LRP  & w/  Sensor Failure: ", r2_score(dropin_predictions_failure, y_test))
    if algorithms_to_execute["LRP"]:
        print("w/  LRP  & w/o Sensor Failure: ", r2_score(dropin_predictions_lrp, y_test))
        print("w/  LRP  & w/  Sensor Failure: ", r2_score(dropin_predictions_failure_lrp, y_test))
        print("w/  LRPr & w/o Sensor Failure: ", r2_score(dropin_predictions_lrp_r, y_test))
        print("w/  LRPr & w/  Sensor Failure: ", r2_score(dropin_predictions_failure_lrp_r, y_test))

    print("MSE Score - DropIn:")
    print("w/o LRP  & w/o Sensor Failure: ", mean_squared_error(dropin_predictions, y_test))
    print("w/o LRP  & w/  Sensor Failure: ", mean_squared_error(dropin_predictions_failure, y_test))
    if algorithms_to_execute["LRP"]:
        print("w/  LRP  & w/o Sensor Failure: ", mean_squared_error(dropin_predictions_lrp, y_test))
        print("w/  LRP  & w/  Sensor Failure: ", mean_squared_error(dropin_predictions_failure_lrp, y_test))
        print("w/  LRPr & w/o Sensor Failure: ", mean_squared_error(dropin_predictions_lrp_r, y_test))
        print("w/  LRPr & w/  Sensor Failure: ", mean_squared_error(dropin_predictions_failure_lrp_r, y_test))

if algorithms_to_execute["SelectiveRetraining"]:
    print("R2 Score - Selective Retraining:")
    print("R2 Score - Without Retraining: ")
    print("           w/o Sensor Failure: ", r2_score(sr_original_predictions, y_test))
    print("           w/  Sensor Failure: ", r2_score(sr_original_predictions_failure, y_test))
    print("R2 Score - Selective Retraining: ")
    print("w/o LRP  & w/o Sensor Failure: ", r2_score(sr_predictions, y_test))
    print("w/o LRP  & w/  Sensor Failure: ", r2_score(sr_predictions_failure, y_test))
    if algorithms_to_execute["LRP"]:
        print("w/  LRP  & w/o Sensor Failure: ", r2_score(sr_predictions_lrp, y_test))
        print("w/  LRP  & w/  Sensor Failure: ", r2_score(sr_predictions_failure_lrp, y_test))

    print("MSE Score - Selective Retraining:")
    print("MSE Score - Without Retraining: ")
    print("           w/o Sensor Failure: ", mean_squared_error(sr_original_predictions, y_test))
    print("           w/  Sensor Failure: ", mean_squared_error(sr_original_predictions_failure, y_test))
    print("MSE Score - Selective Retraining: ")
    print("w/o LRP  & w/o Sensor Failure: ", mean_squared_error(sr_predictions, y_test))
    print("w/o LRP  & w/  Sensor Failure: ", mean_squared_error(sr_predictions_failure, y_test))
    if algorithms_to_execute["LRP"]:
        print("w/  LRP  & w/o Sensor Failure: ", mean_squared_error(sr_predictions_lrp, y_test))
        print("w/  LRP  & w/  Sensor Failure: ", mean_squared_error(sr_predictions_failure_lrp, y_test))
