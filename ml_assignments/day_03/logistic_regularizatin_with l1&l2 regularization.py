
"""
ASSIGNMENT: Logistic Regression with L1 and L2 Regularization
--------------------------------------------------------------

This script implements logistic regression using gradient descent and
extends it by adding L1 (Lasso) and L2 (Ridge) regularization.

The goal is to study how regularization affects the learned parameter
vector beta (β).

--------------------------------------------------------------
MODEL
--------------------------------------------------------------

Logistic regression model:

    z = Xβ
    y_hat = sigmoid(z)

Sigmoid function:

    sigmoid(z) = 1 / (1 + e^(-z))

--------------------------------------------------------------
ORIGINAL COST FUNCTION
--------------------------------------------------------------

Binary Cross Entropy:

J(β) = -(1/n) Σ [ y log(y_hat) + (1-y) log(1-y_hat) ]

--------------------------------------------------------------
L2 REGULARIZATION (Ridge)
--------------------------------------------------------------

Penalty term:

    λ Σ β_j^2

New cost:

J(β) = BCE + λ Σ β_j^2

Gradient becomes:

∂J/∂β = (1/n) Xᵀ(y_hat - y) + 2λβ

Effect:
- Shrinks coefficients
- Reduces overfitting
- Keeps all features

--------------------------------------------------------------
L1 REGULARIZATION (Lasso)
--------------------------------------------------------------

Penalty term:

    λ Σ |β_j|

Gradient approximation:

∂J/∂β = (1/n) Xᵀ(y_hat - y) + λ sign(β)

Effect:
- Forces some coefficients to zero
- Performs feature selection


We test different λ values:

λ = 0
λ = 0.01
λ = 0.1
λ = 1

Observation:
- As λ increases, β values shrink
- Very large λ may cause underfitting

"""

import numpy as np


# --------------------------------------------------------------
# Sigmoid Function
# --------------------------------------------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# --------------------------------------------------------------
# Logistic Regression without Regularization
# --------------------------------------------------------------

def logistic_regression(X, Y, lr=0.1, iterations=1000):

    n, m = X.shape

    X = np.hstack((np.ones((n,1)), X))

    beta = np.random.randn(m+1,1)

    for _ in range(iterations):

        z = X @ beta
        y_hat = sigmoid(z)

        gradient = (1/n) * (X.T @ (y_hat - Y))

        beta = beta - lr * gradient

    return beta


# --------------------------------------------------------------
# Logistic Regression with L2 Regularization
# --------------------------------------------------------------

def logistic_l2(X, Y, lam=0.1, lr=0.1, iterations=1000):

    n, m = X.shape

    X = np.hstack((np.ones((n,1)), X))

    beta = np.random.randn(m+1,1)

    for _ in range(iterations):

        z = X @ beta
        y_hat = sigmoid(z)

        gradient = (1/n) * (X.T @ (y_hat - Y)) + 2 * lam * beta

        beta = beta - lr * gradient

    return beta


# --------------------------------------------------------------
# Logistic Regression with L1 Regularization
# --------------------------------------------------------------

def logistic_l1(X, Y, lam=0.1, lr=0.1, iterations=1000):

    n, m = X.shape

    X = np.hstack((np.ones((n,1)), X))

    beta = np.random.randn(m+1,1)

    for _ in range(iterations):

        z = X @ beta
        y_hat = sigmoid(z)

        gradient = (1/n) * (X.T @ (y_hat - Y)) + lam * np.sign(beta)

        beta = beta - lr * gradient

    return beta


# --------------------------------------------------------------
# Generate Synthetic Dataset
# --------------------------------------------------------------

def generate_data(n=200):

    np.random.seed(42)

    X = np.random.randn(n,2)

    true_beta = np.array([[1.5],[-2],[1]])

    X_intercept = np.hstack((np.ones((n,1)), X))

    probs = sigmoid(X_intercept @ true_beta)

    Y = (probs > 0.5).astype(int)

    return X,Y,true_beta


# --------------------------------------------------------------
# Run Experiment
# --------------------------------------------------------------

if __name__ == "__main__":

    X,Y,true_beta = generate_data()

    print("True Beta:", true_beta.flatten())

    print("\n--- No Regularization ---")
    beta = logistic_regression(X,Y)
    print(beta.flatten())

    lambdas = [0.01,0.1,1]

    print("\n--- L2 Regularization ---")
    for lam in lambdas:
        beta = logistic_l2(X,Y,lam)
        print("lambda =",lam," beta =",beta.flatten())

    print("\n--- L1 Regularization ---")
    for lam in lambdas:
        beta = logistic_l1(X,Y,lam)
        print("lambda =",lam," beta =",beta.flatten())
