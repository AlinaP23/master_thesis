from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.utils import shuffle


def get_dataset(data_set):
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

        activation = 'relu'
        labels = [0, 1, 2]

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
        income = pd.read_csv('./data/adult.data.csv', delimiter=", ")
        X = income[['age', 'workclass', 'fnlwgt', 'education-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']] \
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
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']] \
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

    return X, Y, activation, labels, data_frame


def get_sensor_failure_test_set(original_test_set, random=False, multi_sensor_failure=False, failure_percentage=0.2, ):
    features = range(0, len(original_test_set[0]))
    x_test_failure = np.copy(original_test_set)

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
