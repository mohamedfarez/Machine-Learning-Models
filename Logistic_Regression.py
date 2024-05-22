"""
This code demonstrates the implementation of logistic regression using gradient descent on a set of random data points.

The code generates 20 random data points with binary labels (0 or 1), and then trains a logistic regression model using gradient descent. The trained model is then used to plot the decision boundary on the data.

The key steps in the code are:
1. Generate random data points and labels.
2. Define the sigmoid function for the logistic regression model.
3. Add a bias term to the input data.
4. Initialize the weights with random values.
5. Perform logistic regression training using gradient descent.
6. Plot the data and the decision boundary.
"""
import numpy as np
import matplotlib.pyplot as plt
 
# Generate 20 random data points
np.random.seed(10)  # Set a seed for reproducibility
x = np.random.rand(20) * 10  # Random inputs between 0 and 10
y = (np.random.rand(20) > 0.5).astype(int)  # Random binary labels (0 or 1)
 
# Define the sigmoid function
def sigmoid(z):
  return 1 / (1 + np.exp(-z))
 
# Add a bias term (x0 = 1) for convenience
x = np.c_[np.ones(20), x]  # Add a column of ones to the input data
 
# Initialize weights with random values
w = np.random.rand(2)
 
# Learning rate
learning_rate = 0.01
 
# Perform logistic regression training using gradient descent
for _ in range(1000):
  # Calculate predicted probabilities
  y_pred = sigmoid(np.dot(x, w))
 
  # Calculate the error
  error = y - y_pred
 
  # Update weights using gradient descent
  w += learning_rate * np.dot(x.T, error)
 
# Plot the data and the decision boundary
plt.scatter(x[:, 1], y, color='blue', label='Data')
 
# Get the line equation from the weights
m = -w[1] / w[0]
b = -w[0] / w[0]
x_line = np.linspace(0, 10, 100)
y_line = m * x_line + b
 
plt.plot(x_line, y_line, color='red', label='Decision Boundary')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Logistic Regression with Random Data')
plt.show()