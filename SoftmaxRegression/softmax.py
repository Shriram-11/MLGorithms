import numpy as np


def softmax(z):
    """
    Compute the softmax of each row of the input matrix z.

    Parameters:
    z (ndarray): Input data, shape (n_samples, n_classes).

    Returns:
    ndarray: Softmax probabilities, same shape as z.
    """
    a = np.exp(z - np.max(z, axis=1, keepdims=True))  # Numerical stability
    return a / np.sum(a, axis=1, keepdims=True)


class MultiClass:
    """
    A simple implementation of a Multiclass Logistic Regression classifier.

    Parameters:
    lr (float): Learning rate for gradient descent.
    n_iters (int): Number of iterations for training.
    tol (float): Tolerance for stopping criteria based on the loss.

    Attributes:
    weights (ndarray): Weights of the model, shape (n_features, n_classes).
    bias (ndarray): Biases of the model, shape (n_classes,).
    """

    def __init__(self, lr=0.01, n_iters=1000, tol=1e-4):
        self.lr = lr
        self.n_iters = n_iters
        self.tol = tol
        self.weights = None
        self.bias = None

    def cross_entropy(self, y, y_pred):
        """
        Compute the cross-entropy loss.

        Parameters:
        y (ndarray): True labels, one-hot encoded, shape (n_samples, n_classes).
        y_pred (ndarray): Predicted probabilities, shape (n_samples, n_classes).

        Returns:
        float: The cross-entropy loss.
        """
        n = y.shape[0]
        # Add small epsilon for numerical stability
        logp = -np.log(y_pred + 1e-9) * y
        loss = np.sum(logp) / n
        return loss

    def accuracy(self, y, y_pred):
        """
        Compute the accuracy of predictions.

        Parameters:
        y (ndarray): True labels, one-hot encoded, shape (n_samples, n_classes).
        y_pred (ndarray): Predicted probabilities, shape (n_samples, n_classes).

        Returns:
        float: Accuracy as a percentage.
        """
        return np.mean(np.argmax(y, axis=1) == np.argmax(y_pred, axis=1))

    def fit(self, X, y):
        """
        Train the Multiclass Logistic Regression model using gradient descent.

        Parameters:
        X (ndarray): Input features, shape (n_samples, n_features).
        y (ndarray): True labels, one-hot encoded, shape (n_samples, n_classes).
        """
        n_samples, n_features = X.shape
        n_classes = y.shape[1]
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)

        for i in range(self.n_iters):
            z = np.dot(X, self.weights) + self.bias
            y_pred = softmax(z)
            loss = self.cross_entropy(y, y_pred)

            # Gradient calculation
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y, axis=0)

            # Update weights and biases
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Check for convergence
            if loss < self.tol:
                print(f'Converged after {i+1} iterations')
                break

        print(f'Final loss: {loss}\tFinal accuracy: {
              self.accuracy(y, y_pred)}')

    def predict(self, X):
        """
        Predict the class labels for given input features.

        Parameters:
        X (ndarray): Input features, shape (n_samples, n_features).

        Returns:
        ndarray: Predicted class labels, shape (n_samples,).
        """
        z = np.dot(X, self.weights) + self.bias
        y_pred = softmax(z)
        return np.argmax(y_pred, axis=1)
