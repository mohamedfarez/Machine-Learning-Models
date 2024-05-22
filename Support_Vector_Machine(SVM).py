"""
This code demonstrates the use of a Support Vector Machine (SVM) classifier to classify a simple 2D dataset.

The code first generates a sample dataset with two classes, represented by the X and y variables. It then splits the dataset into training and testing sets using the `train_test_split` function from scikit-learn.

Next, the code creates an SVM classifier with a linear kernel and trains it on the training data using the `fit` method. It then makes predictions on the test data using the `predict` method and prints the predicted and actual labels.

Finally, the code prints some additional information about the trained SVM model, including the support vectors, support vector indices, and the number of support vectors for each class.
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
 
# Generate sample data
X = np.array([[1, 1], [2, 1], [3, 2], [1, 4], [2, 4], [3, 5]])
y = np.array([0, 0, 0, 1, 1, 1])
 
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
 
# Create and train the SVM model
clf = SVC(kernel='linear')  # Use linear kernel for this example
clf.fit(X_train, y_train)
 
# Make predictions on the test set
y_pred = clf.predict(X_test)
 
# Print the predicted labels and actual labels
print("Predicted Labels:", y_pred)
print("Actual Labels:", y_test)
 
# Additional information about the model (optional)
print("Support vectors:", clf.support_vectors_)
print("Support vector indices:", clf.support_)
print("Number of support vectors for each class:", clf.n_support_)
