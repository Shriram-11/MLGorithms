import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
# Replace with the actual module name if necessary
from softmax import MultiClass

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)  # Reshape to a 2D array

# One-hot encode the target labels
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Instantiate the model
model = MultiClass(lr=0.01, n_iters=1000, tol=1e-4)

# Train the model
model.fit(X_train, y_train)

# Test the model on the test set
predictions = model.predict(X_test)

# Convert predictions and true labels back to original class labels
predicted_classes = predictions
true_classes = np.argmax(y_test, axis=1)

# Calculate and print accuracy
accuracy = np.mean(predicted_classes == true_classes)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Output the predicted and true labels for comparison
print("Predicted Classes:", predicted_classes)
print("True Classes:", true_classes)
