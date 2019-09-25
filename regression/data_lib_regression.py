import numpy as np
from sklearn.datasets import make_regression
import pandas as pd


def get_data_set(data_set, n_samples=100, n_features=100, n_informative=10, n_targets=1, bias=0.0, effective_rank=None,
                 tail_strength=0.5, noise=0.0, coef=False, data_shuffle=True, random_state=7):
    """Returns the requested data set.

    Parameters
    ----------
    data_set : String
        Name of the requested data set.
    n_samples : int, optional (default=100)
        The number of samples.
    n_features : int, optional (default=100)
        The total number of features.
    n_informative : int, optional (default=10)
        The number of informative features, i.e., the number of features used to build the linear model used to generate
        the output.
    n_targets : int, optional (default=1)
        The number of regression targets, i.e., the dimension of the y output vector associated with a sample. By
        default, the output is a scalar.
    bias : float, optional (default=0.0)
        The bias term in the underlying linear model.
    effective_rank : int or None, optional (default=None)
        if not None: The approximate number of singular vectors required to explain most of the input data by linear
        combinations. Using this kind of singular spectrum in the input allows the generator to reproduce the
        correlations often observed in practice.
        if None: The input set is well conditioned, centered and gaussian with unit variance.
    tail_strength: float between 0.0 and 1.0, optional (default=0.5)
        The relative importance of the fat noisy tail of the singular values profile if effective_rank is not None.
    noise: float, optional (default=0.0)
        The standard deviation of the gaussian noise applied to the output.
    data_shuffle : boolean, optional (default=True)
        Shuffle the samples and the features.
    coef: boolean, optional (default=False)
        If True, the coefficients of the underlying linear model are returned.
    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset creation. Pass an int for reproducible output across multiple
        function calls.

    Returns
    ----------
    X : array
        The (generated) samples.
    Y : array
        The target values.
    coef: array of shape [n_features] or [n_features, n_targets], optional
        The coefficient of the underlying linear model. It is returned only if coef is True.
    activation: String
        Activation function to be used in a neural network when predicting the target values.
    probabilities: array
        Sensor failure probability for each individual attribute
    """
    activation = 'logistic'
    np.random.seed(5)
    if data_set == "energy_efficiency":
        sensor_data = pd.read_excel('../data/Energy_Efficiency/ENB.xlsx', sheet_name='Sheet 1')

        X = sensor_data.iloc[:, :8].values
        Y = sensor_data.iloc[:, 9].values  # alternatively 8 or 9

        activation = "relu"
        probabilities = np.random.random(len(X[0]))

    elif data_set == "forest_fires":
        sensor_data = pd.read_csv('../data/Forest Fires/forestfires.csv', delimiter=",")

        X = sensor_data.iloc[:, :12].values
        Y = sensor_data.iloc[:, 12].values

        activation = "logistic"
        probabilities = np.random.random(len(X[0]))

    elif data_set == "sklearn":
        X, Y = make_regression(n_samples=n_samples,
                               n_features=n_features,
                               n_informative=n_informative,
                               n_targets=n_targets,
                               bias=bias,
                               effective_rank=effective_rank,
                               tail_strength=tail_strength,
                               noise=noise,
                               shuffle=data_shuffle,
                               coef=coef,
                               random_state=random_state)
        activation = 'logistic'
        probabilities = np.random.random(n_features)

    return X, Y, activation, probabilities


def get_sensor_failure_test_set(original_test_set, np_seed, random=False, multi_sensor_failure=False,
                                failure_percentage=0.2, probability_known=False, probabilities=None):
    """Simulates sensor failure within the test set that is given as input.
    Parameters
    ----------
    original_test_set: array
       Original samples without missing data.
    np_seed: integer
        Seed to make numpy randomization reproducible.
    random: boolean
        If False: equal distribution of "sensor failures" is ensured, i.e. equal percentage of missing data for each
        feature
    multi_sensor_failure: boolean
        If True: failure of a random number of sensors is simulated; if False: failure of one sensor per sample only
    failure_percentage: float
        Determines the percentage of missing data per feature (if random = False)
    probability_known: boolean
        If True: sensor failure probability is given (probabilities)
    probabilities: array
        Determines the sensor failure probabilities (if known)

    Returns
    ----------
    x_test_failure: array
        Samples including simulated sensor failure, i.e. with missing features
    """
    features = range(0, len(original_test_set[0]))
    x_test_failure = np.copy(original_test_set)
    np.random.seed(np_seed)

    # --- MULTI-SENSOR FAILURE --- #
    # simulate random failure of random number of sensors
    if multi_sensor_failure and random and not probability_known:
        for i in range(0, len(original_test_set)):
            no_failing_sensors = np.random.randint(1, len(features))
            sensor_failure = np.random.choice(features, no_failing_sensors, replace=False).tolist()
            x_test_failure[i, sensor_failure] = 0

    # ensure equal distribution of "sensor failures", i.e. equal percentage of missing data for each feature
    elif multi_sensor_failure and not random and not probability_known:
        size_missing_data = int(len(original_test_set) * failure_percentage)
        for i in features:
            missing_data = np.random.choice(range(0, len(original_test_set)), size_missing_data,
                                            replace=False).tolist()
            x_test_failure[missing_data, i] = 0

    # --- SINGLE SENSOR FAILURE --- #
    # simulate random failure of one sensor per tuple
    elif not multi_sensor_failure and random and not probability_known:
        for i in range(0, len(original_test_set)):
            sensor_failure = np.random.choice(features, 1, replace=False).tolist()
            x_test_failure[i, sensor_failure] = 0

    # ensure equal distribution of "sensor failures", i.e. equal percentage of missing data for each feature
    elif not multi_sensor_failure and not random and not probability_known:
        missing_data = list(range(0, len(original_test_set)))
        np.random.shuffle(missing_data)
        tuples_per_feature = int(len(original_test_set) / len(features))
        for i in features:
            start = i * tuples_per_feature + 1
            end = start + tuples_per_feature
            x_test_failure[missing_data[start:end], i] = 0

    # --- FAILURE PROBABILITY KNOWN --- #
    elif probability_known:
        for i in range(0, len(original_test_set)):
            sensor_failure = np.where(np.random.binomial(1, probabilities) == 0)[0]
            x_test_failure[i, sensor_failure] = 0

    return x_test_failure
