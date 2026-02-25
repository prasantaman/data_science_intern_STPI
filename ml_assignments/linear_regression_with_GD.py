import numpy as np

def gradient_descent_linear_regression(X, y, k, tau, lam):
    
    n, m = X.shape
    
    # Add intercept column (xi0 = 1)
    intercept = np.ones((n, 1))
    X = np.hstack((intercept, X))  # Now n x (m+1)
    
    # Initialize beta randomly (m+1 x 1)
    beta = np.random.randn(m + 1, 1)
    
    previous_cost = float('inf')
    
    for _ in range(k):
        
        # Prediction
        y_pred = X @ beta
        
        # Compute Cost (Mean Squared Error)
        errors = y_pred - y
        cost = (1 / (2 * n)) * np.sum(errors ** 2)
        
        # Check stopping condition
        if abs(previous_cost - cost) < tau:
            break
        
        # Gradient computation
        gradient = (1 / n) * (X.T @ errors)
        
        # Update beta
        beta = beta - lam * gradient
        
        previous_cost = cost
    
    return beta, cost
# Generate random data
np.random.seed(42)
X = np.random.randn(10, 3)
y = np.random.randn(10, 1)

beta, final_cost = gradient_descent_linear_regression(
    X, y, k=100, tau=1e-6, lam=0.01
)

print("Learned beta:\n", beta)
print("Final cost:", final_cost)
