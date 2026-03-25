import numpy as np


def logistic_regression_gd(X, Y, k, tau, lr):

    n, m = X.shape

    # Add intercept column
    X = np.hstack((np.ones((n, 1)), X))

    # Initialize beta randomly
    beta = np.random.randn(m + 1, 1)

    prev_cost = float('inf')

    for i in range(k):

        # Linear combination
        z = X @ beta

        # Sigmoid function
        y_hat = 1 / (1 + np.exp(-z))

        # Cost function
        cost = -(1/n) * np.sum(
            Y * np.log(y_hat) + (1 - Y) * np.log(1 - y_hat)
        )

        # Gradient
        gradient = (1/n) * (X.T @ (y_hat - Y))

        # Update beta
        beta = beta - lr * gradient

        # Check stopping condition
        if abs(prev_cost - cost) < tau:
            break

        prev_cost = cost

    return beta, cost
if __name__ == "__main__":

    X = np.random.randn(100,3)
    Y = (np.random.rand(100,1) > 0.5).astype(int)

    beta, final_cost = logistic_regression_gd(
        X,
        Y,
        k=1000,
        tau=1e-6,
        lr=0.01
    )

    print("Learned Coefficients:")
    print(beta)

    print("Final Cost:", final_cost)