from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.datasets import make_classification


def get_data_set(data_set, n_samples=100, n_features=20, n_informative=2, n_redundant=2, n_repeated=0, n_classes=2,
                 n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0,
                 data_shuffle=True, random_state=7):
    """Returns the requested data set.

    Parameters
    ----------
    data_set : String
        Name of the requested data set.
    n_samples : int, optional (default=100)
        The number of samples.
    n_features : int, optional (default=20)
        The total number of features. These comprise n_informative informative features, n_redundant redundant features,
        n_repeated duplicated features and n_features-n_informative-n_redundant-n_repeated useless features drawn at
        random.
    n_informative : int, optional (default=2)
        The number of informative features. Each class is composed of a number of gaussian clusters each located around
        the vertices of a hypercube in a subspace of dimension n_informative. For each cluster, informative features are
        drawn independently from N(0, 1) and then randomly linearly combined within each cluster in order to add
        covariance. The clusters are then placed on the vertices of the hypercube.
    n_redundant : int, optional (default=2)
        The number of redundant features. These features are generated as random linear combinations of the informative
        features.
    n_repeated : int, optional (default=0)
        The number of duplicated features, drawn randomly from the informative and the redundant features.
    n_classes : int, optional (default=2)
        The number of classes (or labels) of the classification problem.
    n_clusters_per_class : int, optional (default=2)
        The number of clusters per class.
    weights : list of floats or None (default=None)
        The proportions of samples assigned to each class. If None, then classes are balanced. Note that if
        len(weights) == n_classes - 1, then the last class weight is automatically inferred. More than n_samples samples
        may be returned if the sum of weights exceeds 1.
    flip_y : float, optional (default=0.01)
        The fraction of samples whose class are randomly exchanged. Larger values introduce noise in the labels and make
        the classification task harder.
    class_sep : float, optional (default=1.0)
        The factor multiplying the hypercube size. Larger values spread out the clusters/classes and make the
        classification task easier.
    hypercube : boolean, optional (default=True)
        If True, the clusters are put on the vertices of a hypercube. If False, the clusters are put on the vertices of
        a random polytope.
    shift : float, array of shape [n_features] or None, optional (default=0.0)
        Shift features by the specified value. If None, then features are shifted by a random value drawn in
        [-class_sep, class_sep].
    scale : float, array of shape [n_features] or None, optional (default=1.0)
        Multiply features by the specified value. If None, then features are scaled by a random value drawn in [1, 100].
        Note that scaling happens after shifting.
    data_shuffle : boolean, optional (default=True)
        Shuffle the samples and the features.
    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset creation. Pass an int for reproducible output across multiple
        function calls.

    Returns
    ----------
    X : array
        The (generated) samples.
    Y : array
        The labels for class membership of each sample.
    activation: String
        Activation function to be used in a neural network when classifying the data set
    labels: Array
        List of the unique labels of the data set
    data_frame: Boolean
        Indicates whether the labels of the data set are represented in binary format
    """
    data_frame = False
    if data_set == "iris":
        iris = pd.read_csv('./data/iris.csv')

        # Create numeric classes for species (0,1,2)
        iris.loc[iris['species'] == 'virginica', 'species'] = 0
        iris.loc[iris['species'] == 'versicolor', 'species'] = 1
        iris.loc[iris['species'] == 'setosa', 'species'] = 2

        # Create Input and Output columns
        X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
        Y = iris[['species']].values.ravel()
        Y = pd.get_dummies(Y)

        activation = 'relu'
        labels = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        data_frame = True

    elif data_set == "bank":
        bank = pd.read_csv('./data/bank_data_balanced.csv', delimiter=";")

        # Create Input and Output columns
        X = bank[['age', 'job_num', 'marital_num', 'education_num', 'default_num', 'housing_num', 'loan_num',
                  'contact_num', 'month_num', 'day_num', 'duration']].values
        # One-hot encode Y so that the NN is created with two output nodes
        Y = bank[['y']].values.ravel()
        Y = pd.get_dummies(Y)

        activation = 'logistic'
        labels = [[1, 0], [0, 1]]
        data_frame = True

    elif data_set == "income":
        # https://archive.ics.uci.edu/ml/datasets/Adult
        income = pd.read_csv('./data/adult.data.csv', delimiter=";")
        X = income[['age', 'workclass', 'fnlwgt', 'education-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']]\
            .values
        label_encoder = LabelEncoder()
        for column in [1, 4, 5, 6, 7, 8, 12]:
            X[:, column] = label_encoder.fit_transform(X[:, column])

        income.loc[income['y'] == '<=50K', 'y'] = 0
        income.loc[income['y'] == '>50K', 'y'] = 1
        Y = income[['y']].values.ravel()

        activation = 'logistic'
        labels = [0, 1]

    elif data_set == "income_balanced":
        # https://archive.ics.uci.edu/ml/datasets/Adult
        income = pd.read_csv('./data/adult.data_balanced.csv', delimiter=";")
        income = shuffle(income)
        X = income[['age', 'workclass', 'fnlwgt', 'education-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']]\
            .values
        label_encoder = LabelEncoder()
        for column in [1, 4, 5, 6, 7, 8, 12]:
            X[:, column] = label_encoder.fit_transform(X[:, column])

        income.loc[income['y'] == '<=50K', 'y'] = 0
        income.loc[income['y'] == '>50K', 'y'] = 1
        Y = income[['y']].values.ravel()
        Y = pd.get_dummies(Y)

        activation = 'logistic'
        labels = [[1, 0], [0, 1]]
        data_frame = True

    elif data_set == "gas_sensor_array_drift":
        # https://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset+at+Different+Concentrations
        sensor_data = pd.read_csv('./data/gas_sensor_array_drift/batch1.dat', delimiter=" ", header=None)
        sensor_data = sensor_data.append(pd.read_csv('./data/gas_sensor_array_drift/batch2.dat', delimiter=" ", header=None))
        sensor_data = sensor_data.append(pd.read_csv('./data/gas_sensor_array_drift/batch3.dat', delimiter=" ", header=None))

        sensor_data = shuffle(sensor_data, random_state=random_state)
        for c in sensor_data.columns.values:
            if c == 0:
                Y = sensor_data[c].apply(lambda x: float(str(x).split(';')[0]))
                sensor_data[c] = sensor_data[c].apply(lambda x: float(str(x).split(';')[1]))
            elif c < 129:
                sensor_data[c] = sensor_data[c].apply(lambda x: float(str(x).split(':')[1]))

        X = sensor_data.iloc[:, 0:129].values

        activation = 'logistic'
        labels = [1, 2, 3, 4, 5, 6]

    elif data_set == "sklearn":
        X, Y = make_classification(n_samples=n_samples,
                                   n_features=n_features,
                                   n_informative=n_informative,
                                   n_redundant=n_redundant,
                                   n_repeated=n_repeated,
                                   n_classes=n_classes,
                                   n_clusters_per_class=n_clusters_per_class,
                                   weights=weights,
                                   flip_y=flip_y,
                                   class_sep=class_sep,
                                   hypercube=hypercube,
                                   shift=shift,
                                   scale=scale,
                                   shuffle=data_shuffle,
                                   random_state=random_state)
        Y = pd.get_dummies(Y)
        activation = 'relu'
        labels = []
        for i in range(0, n_classes):
            single_label = [0] * n_classes
            single_label[i] = 1
            labels.append(single_label)
        data_frame = True

    return X, Y, activation, labels, data_frame


def get_sensor_failure_test_set(original_test_set, np_seed, random=False, multi_sensor_failure=False,
                                failure_percentage=0.2):
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
    if multi_sensor_failure and random:
        for i in range(0, len(original_test_set)):
            no_failing_sensors = np.random.randint(1, len(features))
            sensor_failure = np.random.choice(features, no_failing_sensors, replace=False).tolist()
            x_test_failure[i, sensor_failure] = 0

    # ensure equal distribution of "sensor failures", i.e. equal percentage of missing data for each feature
    elif multi_sensor_failure and not random:
        size_missing_data = int(len(original_test_set) * failure_percentage)
        for i in features:
            missing_data = np.random.choice(range(0, len(original_test_set)), size_missing_data,
                                            replace=False).tolist()
            x_test_failure[missing_data, i] = 0

    # --- SINGLE SENSOR FAILURE --- #
    # simulate random failure of one sensor per tuple
    elif not multi_sensor_failure and random:
        for i in range(0, len(original_test_set)):
            sensor_failure = np.random.choice(features, 1, replace=False).tolist()
            x_test_failure[i, sensor_failure] = 0

    # ensure equal distribution of "sensor failures", i.e. equal percentage of missing data for each feature
    elif not multi_sensor_failure and not random:
        missing_data = list(range(0, len(original_test_set)))
        np.random.shuffle(missing_data)
        tuples_per_feature = int(len(original_test_set) / len(features))
        for i in features:
            start = i * tuples_per_feature + 1
            end = start + tuples_per_feature
            x_test_failure[missing_data[start:end], i] = 0

    return x_test_failure
