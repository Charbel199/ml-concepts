from sklearn.svm import SVR  # Support Vector Regression model
from sklearn import datasets # To load example datasets
import matplotlib.pyplot as plt # For visualizing the data

# Load the digits dataset
# The digits dataset contains 8x8 images of hand-written digits (0-9) and their corresponding labels.
digits = datasets.load_digits()

# Initialize the Support Vector Regression (SVR) model
# Parameters:
# - kernel='rbf': Specifies the Radial Basis Function kernel, suitable for nonlinear regression
# - C=100: Regularization parameter; higher values reduce the margin but may lead to overfitting
# - epsilon=0.1: Defines a margin of tolerance where predictions are not penalized
model = SVR(kernel='rbf', C=100, epsilon=0.1)
# Train the model using all but the last data point in the dataset
model.fit(digits.data[:-1], digits.target[:-1])

# Predict the digit for the last image in the dataset
# Note: SVR performs regression, so the output will be a continuous value
pred = model.predict(digits.data[-1:])

# Visualize the last image in the dataset
plt.matshow(digits.images[-1])  # digits.images contains the raw image data in matrix form
plt.show()                      # Display the image

# Print the predicted digit
# Since SVR is used, the predicted value may not be an integer, but rather a continuous approximation.
print(f"Predicted digit: {pred}")
