import numpy as np

def SGD(X, y, learning_rate=0.01, epochs=100,tol=1e-4):
    # Initialize weights to zeros
    weights = np.zeros(X.shape[1])
    loss=float('inf')

    # Loop through the specified number of epochs
    for e in range(epochs):
        # Loop through each data point (stochastic gradient descent)
        for i in range(X.shape[0]):
            # Compute the gradient for the current data point
            gradient = -X[i] * (y[i] - np.dot(X[i], weights))
            
            # Update weights using the gradient and learning rate
            weights = weights - learning_rate * gradient
        # Calculate the loss after the epoch
        nloss = np.mean((np.dot(X, weights) - y)**2)

        # Check for convergence
        if abs(loss - nloss) < tol:
            print(f"Converged after {e} epochs.")
            break

        loss = loss-nloss

    # Return the final weights
    return weights
