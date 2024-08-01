import numpy as np
# Adjust this import statement according to the actual file name
from k_nearest_neighbours import KNN

# Define dummy data
X_train = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6]
])

y_train = np.array([0, 0, 1, 1, 1])

X_test = np.array([
    [1, 2],
    [4, 5]
])

y_test = np.array([0, 1])

# Create KNN instance
k = 3
knn = KNN(X_train, y_train, k)

# Predict on the test set
predictions = [knn.predict(x) for x in X_test]

# Calculate accuracy
accuracy = np.mean(predictions == y_test)

print("Predictions:", predictions)
print("Actual values:", y_test)
print("Accuracy:", accuracy)
