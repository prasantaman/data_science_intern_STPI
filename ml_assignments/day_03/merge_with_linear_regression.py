# Merge the linear regression code base created in Machine learning Exercise 1 and the logistic regression code base created in this Exercise 
# and create an object oriented code base that maximises reuse of code across the algorithms.

import numpy as np


class BaseModel:

    def __init__(self, lr=0.01, iterations=1000):
        self.lr = lr
        self.iterations = iterations
        self.beta = None


    def add_intercept(self, X):
        n = X.shape[0]
        return np.hstack((np.ones((n,1)), X))


    def fit(self, X, y):

        X = self.add_intercept(X)
        n, m = X.shape

        self.beta = np.random.randn(m,1)

        for _ in range(self.iterations):

            y_pred = self.predict_internal(X)

            gradient = (1/n) * (X.T @ (y_pred - y))

            self.beta = self.beta - self.lr * gradient


    def predict_internal(self, X):
        raise NotImplementedError


    def predict(self, X):
        X = self.add_intercept(X)
        return self.predict_internal(X)


class LinearRegression(BaseModel):

    def predict_internal(self, X):
        return X @ self.beta

class LogisticRegression(BaseModel):

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


    def predict_internal(self, X):
        z = X @ self.beta
        return self.sigmoid(z)


    def predict_class(self, X):
        probs = self.predict(X)
        return (probs >= 0.5).astype(int)
X = np.random.randn(100,2)
y = np.random.randn(100,1)

model = LinearRegression()
model.fit(X,y)

pred = model.predict(X)
model = LogisticRegression()
model.fit(X,y)
print(model.beta)
print(model.predict(X))
print(model.predict_class(X))
print(pred)
pred = model.predict_class(X)    