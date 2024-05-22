"""
This code demonstrates the use of a Random Forest Classifier to predict weather conditions based on simulated weather data.

The code performs the following steps:
1. Generates random weather data with 5 features (Temperature, Humidity, Wind Speed, Pressure, Cloud Cover) and 100 data points.
2. Creates labels for the data (sunny, rainy, cloudy) using random choice.
3. Splits the data into features (X) and target variable (y).
4. Creates a Random Forest Classifier model with 100 estimators and a random state of 42.
5. Trains the model on the data.
6. Generates new data for prediction and uses the trained model to predict the weather.
7. Prints the features and the predicted weather.
8. Calculates the feature importances using permutation importance and plots the results.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
 
# Generate random weather data
np.random.seed(42)
data = np.random.rand(100, 5)  # 100 data points, 5 features
 
# Define feature names
features = ["Temperature", "Humidity", "Wind Speed", "Pressure", "Cloud Cover"]
 
# Create labels (sunny, rainy, cloudy)
labels = np.random.choice(["sunny", "rainy", "cloudy"], size=100)
 
# Split data into features and target variable
X = data
y = labels
 
# Create a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
 
# Train the model
model.fit(X, y)
 
# Generate new data for prediction
new_data = np.random.rand(1, 5)
 
# Predict the weather for the new data
prediction = model.predict(new_data)[0]
 
# Print the features and predicted weather
print("Features:", new_data[0])
print("Predicted weather:", prediction)
 
# Calculate feature importances
result = permutation_importance(model, X, y, n_repeats=10)
importances = result.importances_mean
 
# Plot feature importances
plt.bar(features, importances)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature importances in Random Forest")
plt.show()