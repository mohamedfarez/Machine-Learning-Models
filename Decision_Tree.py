"""
This code demonstrates the use of a Decision Tree Classifier from the scikit-learn library to make predictions on a simple dataset.

The code first creates a synthetic dataset with 3 features and 2 classes using the `make_blobs` function. It then creates a Decision Tree Classifier, trains it on the dataset, and makes a prediction on a new data point.

The predicted class is printed to the console.
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_blobs
 
# Create a dataset with 3 features and 2 classes
X, y = make_blobs(n_samples=1000, centers=2, n_features=3, random_state=0)
 
# Create a decision tree classifier
clf = DecisionTreeClassifier()
 
# Train the classifier on the data
clf.fit(X, y)
 
# Make predictions on a new data point
new_data = [[5, 3, 1]]  # Example data point
prediction = clf.predict(new_data)
 
# Print the predicted class
print("Predicted class:", prediction[0])