import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def knn_imputation(x_test_failure, n_neighbors):
    x_test_knn_imputation = np.copy(x_test_failure)
    x_test_failure = np.array(x_test_failure)
    # imputation via kNN
    for i in range(len(x_test_failure)):
        missing_data = np.where(x_test_failure[i] == 0)[0]
        for j in missing_data:
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            # find indices of all tuples missing attribute j -> they should not be used within kNN
            indices = np.where(x_test_failure.T[j] == 0)[0]
            X = np.delete(np.delete(np.copy(x_test_failure), j, axis=1), indices, axis=0)
            y = np.delete(np.copy(x_test_failure.T[j]), indices)
            knn.fit(X, y)
            x_test_knn_imputation[i][j] = knn.predict([np.delete(np.copy(x_test_failure[i]), j)])
    return x_test_knn_imputation


def mean_imputation(x_test_failure):
    x_test_mean_imputation = np.copy(x_test_failure)
    means = []

    # calculate mean for each variable (ignoring missing data)
    x_test_failure = np.array(x_test_failure)
    for column in x_test_failure.T:
        column = np.extract(column != 0, column)
        means.append(np.mean(column))

    # imputation: replace missing data with means
    for i in range(len(x_test_mean_imputation)):
        missing_data = np.where(x_test_mean_imputation[i] == 0)[0]
        for j in missing_data:
            x_test_mean_imputation[i][j] = means[j]

    return x_test_mean_imputation

def median_imputation(x_test_failure):
    x_test_median_imputation = np.copy(x_test_failure)
    medians = []

    # calculate median for each variable (ignoring missing data)
    x_test_failure = np.array(x_test_failure)
    for column in x_test_failure.T:
        column = np.extract(column != 0, column)
        medians.append(np.median(column))

    # imputation: replace missing data with medians
    for i in range(len(x_test_median_imputation)):
        missing_data = np.where(x_test_median_imputation[i] == 0)[0]
        for j in missing_data:
            x_test_median_imputation[i][j] = medians[j]

    return x_test_median_imputation

x_f = [[1,2,3,0],
       [0,4,0,2],
       [3,0,5,1],
       [3,4,0,3]]
print(mean_imputation(x_f))
print(median_imputation(x_f))
print(knn_imputation(x_f, 2))