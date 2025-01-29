#Write a program to Implement the non-parametric Locally Weighted Regression algorithm in order to fit data points. Select appropriate data set for your experiment and draw graphs
import numpy as np
import matplotlib.pyplot as plt

# Locally Weighted Regression function
def lwr(x_query, X, y, tau):
    weights = np.exp(-np.sum((X - x_query) ** 2, axis=1) / (2 * tau ** 2))
    W = np.diag(weights)
    
    # Add bias term to feature matrix
    X_bias = np.c_[np.ones(X.shape[0]), X]
    theta = np.linalg.inv(X_bias.T @ W @ X_bias) @ X_bias.T @ W @ y
    
    return np.array([1, x_query]) @ theta  # Return prediction for x_query

# Generate synthetic data
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * np.sin(X).ravel() + np.random.normal(scale=0.5, size=X.shape[0])

# Make predictions
tau = 0.5  # Bandwidth parameter
X_test = np.linspace(0, 10, 100)
y_pred = np.array([lwr(x, X, y, tau) for x in X_test])

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Training Data')
plt.plot(X_test, y_pred, color='red', label='LWR Prediction')
plt.title('Locally Weighted Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
