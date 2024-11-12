import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Load the dataset
data = load_iris()
X = data.data
y = data.target

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter values to test
solvers = ['lbfgs', 'liblinear']
C_values = [0.1, 1.0, 10]

# Loop through the combinations of hyperparameters
for solver in solvers:
    for C in C_values:
        with mlflow.start_run():
            # Set up the model with current hyperparameters
            model = LogisticRegression(solver=solver, C=C, max_iter=200)

            # Log parameters
            mlflow.log_param("solver", solver)
            mlflow.log_param("C", C)
            mlflow.log_param("max_iter", 200)

            # Train the model
            model.fit(X_train, y_train)

            # Make predictions and calculate metrics
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)

            # Log the model
            mlflow.sklearn.log_model(model, "logistic_regression_model")

            # Print results for each run
            print(f"Run with solver={solver}, C={C}, accuracy={accuracy}")

print("Experiments complete. Check MLflow UI for comparison.")
