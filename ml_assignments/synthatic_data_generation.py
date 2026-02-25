import numpy as np

def generate_linear_data(sigma, n, m):
    # Step 1: Generate random independent variables (n x m)
    x_random = np.random.randn(n, m)
    
    # Step 2: Add intercept column (xi0 = 1)
    intercept = np.ones((n, 1))
    x = np.hstack((intercept, x_random))  # n x (m+1)
    
    # Step 3: Generate random beta coefficients (m+1 x 1)
    beta = np.random.randn(m + 1, 1)
    
    # Step 4: Generate Gaussian noise (n x 1)
    noise = np.random.normal(0, sigma, (n, 1))
    
    # Step 5: Compute y = Xβ + e
    y = x @ beta + noise
    
    return x, y, beta
print(generate_linear_data(sigma=1.0, n=10, m=5))
