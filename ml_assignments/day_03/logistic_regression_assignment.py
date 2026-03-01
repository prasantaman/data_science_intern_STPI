# Task 1:    Write a function to generate an m+1 dimensional data set of size n, consisting of m continuous independent
# variables (X) and one dependent binary variable (Y), defined as 

# Y = 1 if p(y = 1 | x) = 1 / (1 + exp(-x · β)) > 0.5, otherwise Y = 0

# Where,
# • β is a random vector of dimensionality m + 1, representing the coefficients of the linear relationship between X and Y, and
# • ∀i ∈ [1, n], xi0 = 1

# To add noise to the labels (Y) generated, we assume a Bernoulli distribution with probability of success, θ, that determines whether or not the label generated, as above, is to be flipped. The larger the value of θ, the greater is the noise.

# The function should take the following parameters:
# • θ: The probability of flipping the label, Y
# • n: The size of the data set
# • m: The number of independent variables

# Output from the function should be:
# • X: An n × m numpy array of independent variable values (with a 1 in the first column)
# • Y: The n × 1 binary numpy array of output values
# • β: The random coefficients used to generate Y from X
import numpy as np 
def generate_logistic_data(theta,n,m):
    beta = np.random.randn(m+1)
    x_random = np.random.randn(n,m)
    ones = np.ones((n,1))
    x = np.hstack((ones,x_random))
    z = np.dot(x,beta)
    p = 1/(1 +np.exp(-z))
    y = (p>0.5).astype(int)
    flip = np.random.binomial(1,theta,n)
    y = np.abs(y-flip)
    y = y.reshape(-1,1)
    return x,y,beta

theta = 0.2   # 20% noise
n = 100       # 100 samples
m = 3         # 3 independent variables

X, Y, beta = generate_logistic_data(theta, n, m)

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Beta:", beta)