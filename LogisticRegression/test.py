import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import datasets

from Logistic import LogisticRegression

import matplotlib.pyplot as plt

data = datasets.load_breast_cancer()

X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=50)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test)/len(y_test)


print(accuracy(y_pred, y_test))
