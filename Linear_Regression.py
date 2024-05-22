"""
This code demonstrates a simple linear regression example using the NumPy library.

The code first generates random data points (x, y) where y = 3x + 2 with some added noise. It then performs a linear regression using the `numpy.polyfit` function to find the slope (m) and intercept (b) of the best-fit line.

The predicted y values are calculated using the regression line equation (y = mx + b), and the data points and regression line are plotted using Matplotlib.

Finally, the plot is displayed with labels, a title, and a legend.
"""
import numpy as np
import matplotlib.pyplot as plt
 
# Generate random data
np.random.seed(10)  # Set a seed for reproducibility
x = np.random.rand(20)
y = 3 * x + 2 + np.random.randn(20)  # Add some noise
 
# Perform linear regression using numpy.polyfit
m, b = np.polyfit(x, y, 1)
 
# Calculate predicted y values
y_pred = m * x + b
 
# Plot the data and regression line
plt.scatter(x, y, color='blue', label='Data')
plt.plot(x, y_pred, color='red', label='Regression Line')
 
# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression with Scatter Plot')
 
# Add legend
plt.legend()
 
# Show the plot
plt.grid(True)
plt.show()