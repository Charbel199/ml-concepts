from sklearn.ensemble import RandomForestClassifier  # Random Forest model
from sklearn import datasets                         # To load example datasets
import matplotlib.pyplot as plt                      # For visualizing the data

# Load the digits dataset
# The digits dataset contains 8x8 images of hand-written digits (0-9) and their corresponding labels.
digits = datasets.load_digits()

# Define the number of trees (estimators) for the Random Forest
n = 100  # Using 100 decision trees
# Initialize the Random Forest Classifier model
model = RandomForestClassifier(n_estimators=n)

# Train the model using all but the last data point in the dataset
model.fit(digits.data[:-1], digits.target[:-1])

# Predict the digit for the last image in the dataset
pred = model.predict(digits.data[-1:])

# Visualize the last image in the dataset
plt.matshow(digits.images[-1])  # digits.images contains the raw image data in matrix form
plt.show()                      # Display the image

# Print the predicted digit
print(f"Predicted digit: {pred}") 
