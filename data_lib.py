from sklearn.preprocessing import LabelEncoder
import pandas as pd


def get_dataset(data_set):
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
        labels = [0, 1]

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

    return X, Y, activation, labels
