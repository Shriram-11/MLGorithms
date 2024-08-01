import numpy as np
# Placeholder for additional code

# This function calculates the Euclidean distance between two points


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a-b)**2))

# KNN class for performing k-nearest neighbors classification


class KNN():
    def __init__(self, X, y, k=2):
        self.k = k
        self.X_train = X
        self.y_train = y
        self.n = len(X)
        self.labels = np.unique(y)

    # Predicts the class label for a given input sample
    def predict(self, x):
        # Calculate the distances between the input sample and all training samples
        distances = [(euclidean_distance(x, self.X_train[i]), i)
                     for i in range(self.n)]

        # Sort the distances in ascending order
        distances.sort(key=lambda x: x[0])

        # Get the indices of the k nearest neighbors
        indices = [distances[i][1] for i in range(self.k)]

        # Get the class labels of the k nearest neighbors
        labels = [self.y_train[i] for i in indices]

        # Count the occurrences of each class label
        c = {l: 0 for l in self.labels}
        for l in labels:
            c[l] += 1

        # Return the class label with the highest count
        return max(c, key=c.get)
