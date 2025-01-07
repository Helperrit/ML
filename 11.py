# Write a program to Implement the non-parametric Locally Weighted Regression algorithm in order to fit data points. Select appropriate data set for your experiment and draw graphs
import numpy as np
import matplotlib.pyplot as plt

# Locally Weighted Regression function
def locally_weighted_regression(x_query, X, y, tau):
    m = X.shape[0]
    # Compute weights using Gaussian kernel
    weights = np.exp(-np.sum((X - x_query) ** 2, axis=1) / (2 * tau ** 2))
    W = np.diag(weights)  # Diagonal weight matrix
    X_b = np.c_[np.ones((m, 1)), X]  # Add bias term (column of ones)
    
    # Compute theta using the Normal Equation
    theta = np.linalg.inv(X_b.T @ W @ X_b) @ (X_b.T @ W @ y)
    
    # Add bias term to the query point
    x_query_b = np.r_[1, x_query]  # Append 1 for bias
    return x_query_b @ theta  # Return prediction for x_query

# Generate synthetic data
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = 2 * np.sin(X) + np.random.normal(scale=0.5, size=X.shape)

X = X.reshape(-1, 1)  # Convert X to a column vector

# Fit the Locally Weighted Regression model
tau = 0.5  # Bandwidth parameter
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred = np.array([locally_weighted_regression(x_query, X, y, tau) for x_query in X_test])

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Training Data')
plt.plot(X_test, y_pred, color='red', label='LWR Prediction')
plt.title('Locally Weighted Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
