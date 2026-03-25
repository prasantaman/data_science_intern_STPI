
"""
LOGISTIC REGRESSION GRADIENT DESCENT REPORT
Author: Student
Description:
This script investigates how different values of sample size (n)
and parameter magnitude (theta) affect the ability of logistic regression
to learn the coefficients beta.

The file contains:

1. Mathematical explanation (in comments)
2. Logistic regression implementation using gradient descent
3. Experiments with different sample sizes
4. Experiments with different theta values
5. Printed comparison between true theta and learned beta

--------------------------------------------------------------------
MATHEMATICAL MODEL
--------------------------------------------------------------------

Logistic regression model:

    z = Xβ
    ŷ = σ(z)

Where sigmoid function:

    σ(z) = 1 / (1 + e^(-z))

Output probability:

    ŷ ∈ (0,1)

Binary prediction:

    if ŷ >= 0.5 → class = 1
    else → class = 0


--------------------------------------------------------------------
COST FUNCTION (Binary Cross Entropy)
--------------------------------------------------------------------

J(β) = -(1/n) Σ [ y log(ŷ) + (1-y) log(1-ŷ) ]

Where:
    y  = actual label
    ŷ  = predicted probability

Goal:
    Minimize J(β)


--------------------------------------------------------------------
PARTIAL DERIVATIVE OF COST FUNCTION
--------------------------------------------------------------------

Using chain rule and derivative of sigmoid:

    dσ/dz = σ(z)(1-σ(z))

After simplification:

    ∂J/∂β = (1/n) Xᵀ (ŷ - y)

This gives gradient direction.

Gradient descent update rule:

    β = β - λ * gradient

Where:
    λ = learning rate


--------------------------------------------------------------------
EXPERIMENT GOAL
--------------------------------------------------------------------

We investigate:

1. Effect of sample size n
2. Effect of parameter magnitude θ

Observation expected:

Large n:
    Better estimation of true parameters

Small n:
    Higher variance in learned parameters

Large θ:
    Steeper sigmoid curve
    Harder optimization sometimes

Small θ:
    Smoother learning


--------------------------------------------------------------------
"""


import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_regression_gd(X, Y, k=1000, tau=1e-6, lr=0.1):
    n, m = X.shape

    # add intercept
    X = np.hstack((np.ones((n,1)), X))

    beta = np.random.randn(m+1,1)

    prev_cost = float("inf")

    for _ in range(k):

        z = X @ beta
        y_hat = sigmoid(z)

        cost = -(1/n) * np.sum(
            Y*np.log(y_hat+1e-10) +
            (1-Y)*np.log(1-y_hat+1e-10)
        )

        if abs(prev_cost - cost) < tau:
            break

        gradient = (1/n) * (X.T @ (y_hat - Y))

        beta = beta - lr * gradient

        prev_cost = cost

    return beta, cost



def generate_dataset(n, theta):
    """Generate synthetic logistic regression dataset"""

    m = len(theta)-1

    X = np.random.randn(n, m)

    X_intercept = np.hstack((np.ones((n,1)), X))

    z = X_intercept @ theta.reshape(-1,1)

    probs = sigmoid(z)

    Y = (probs > 0.5).astype(int)

    return X, Y



def run_experiment_sample_size():
    print("\n---- Experiment: Effect of sample size (n) ----\n")


    theta_true = np.array([1.5, -2.0])

    sample_sizes = [20, 100, 1000]

    for n in sample_sizes:

        X,Y = generate_dataset(n, theta_true)

        beta_learned, cost = logistic_regression_gd(X,Y)

        print("Sample size:", n)
        print("True theta:", theta_true)
        print("Learned beta:", beta_learned.flatten())
        print("Final cost:", cost)
        print("------------------------------------")



def run_experiment_theta():
    print("\n---- Experiment: Effect of theta magnitude ----\n")


    theta_values = [
        np.array([0.5, -0.5]),
        np.array([2.0, -2.0]),
        np.array([5.0, -5.0])
    ]

    n = 500

    for theta in theta_values:

        X,Y = generate_dataset(n, theta)

        beta_learned, cost = logistic_regression_gd(X,Y)

        print("True theta:", theta)
        print("Learned beta:", beta_learned.flatten())
        print("Final cost:", cost)
        print("------------------------------------")



if __name__ == "__main__":

    np.random.seed(42)

    run_experiment_sample_size()

    run_experiment_theta()

