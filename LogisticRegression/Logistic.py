import numpy as np

'''
h(x)= 1/(1+e^(-(wx+b)))

Cost function
J=[1/N] * summation 1 to n[yi log(h(xi)) + (1-yi)log(1-h(xi))]

Gradient

dw = [1/N] summation 2xi(y_pred-yi)
db= [1/n] summation 2(y_pred-yi)
'''


def sigmoid(x):
    return 1/(1+np.exp(-x))


class LogisticRegression():
    def __init__(self, lr=0.001, n_iters=100):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        samples, features = X.shape
        self.bias = 0.0
        self.weights = np.zeros(features)

        for _ in range(self.n_iters):
            lin = np.dot(X, self.weights)+self.bias
            y_pred = sigmoid(lin)

            dw = (1/samples)*(np.dot(X.T, y_pred-y))
            db = (1/samples)*(np.sum(y_pred-y))

            self.weights -= self.lr*dw
            self.bias -= self.lr*db

    def predict(self, X):
        lin = np.dot(X, self.weights)+self.bias
        res = sigmoid(lin)

        return [0 if y < 0.5 else 1 for y in res]
