import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
# Generate a simple sine wave as our time series data
data = np.sin(np.linspace(0, 100, 1000))
scaler = MinMaxScaler()
data = scaler.fit_transform(data.reshape(-1, 1)).reshape(-1)
sequence_length = 50
batch_size=128

# -------------------------- PREDICTION --------------------------
# Prepare input-output pairs for prediction training
def create_sequences_for_prediction(data, sequence_length):
    sequences = []
    labels = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
        labels.append(data[i + sequence_length])
    return np.array(sequences), np.array(labels)

X_pred, y_pred = create_sequences_for_prediction(data, sequence_length)
X_pred = torch.tensor(X_pred, dtype=torch.float32)
y_pred = torch.tensor(y_pred, dtype=torch.float32)
dataset_pred = TensorDataset(X_pred, y_pred)
dataloader_pred = DataLoader(dataset_pred, batch_size=batch_size, shuffle=True)

# Define LSTM for prediction
class LSTMModelPredict(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMModelPredict, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the last time step's output
        return out

# Training the prediction model
model_pred = LSTMModelPredict()
criterion_pred = nn.MSELoss()
optimizer_pred = torch.optim.Adam(model_pred.parameters(), lr=0.001)
num_epochs = 30

for epoch in range(num_epochs):
    loop = tqdm(dataloader_pred, leave=True)
    for inputs, targets in loop:
        inputs = inputs.unsqueeze(-1)  # Add feature dimension
        outputs = model_pred(inputs)
        loss = criterion_pred(outputs.squeeze(), targets)

        optimizer_pred.zero_grad()
        loss.backward()
        optimizer_pred.step()

        # Update tqdm progress bar
        loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
        loop.set_postfix(
            loss=loss.item()
        )


# Prepare the last sequence to predict the next value
test_seq_pred = torch.tensor(data[-sequence_length:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
model_pred.eval()
with torch.no_grad():
    predicted_value = model_pred(test_seq_pred).item()

# Display prediction result
print(f'Next value prediction: {predicted_value}')
plt.plot(data, label="Original Data")
plt.axvline(x=len(data) - sequence_length, color="r", linestyle="--", label="Prediction Point")
plt.scatter(len(data), predicted_value, color="green", label="Predicted Value")
plt.legend()
plt.show()

# -------------------------- CLASSIFICATION --------------------------
# Convert the data into binary labels for classification
# Threshold to classify data points
threshold = 0.2
def create_sequences_for_classification(data, sequence_length):
    sequences = []
    labels = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
        labels.append(1 if (sum(data[i:i + sequence_length]) / len(data[i:i + sequence_length])) > threshold else 0) # Classify based on average
    return np.array(sequences), np.array(labels)

X_class, y_class = create_sequences_for_classification(data, sequence_length)
X_class = torch.tensor(X_class, dtype=torch.float32)
y_class = torch.tensor(y_class, dtype=torch.long)  # Classification labels should be long tensors
dataset_class = TensorDataset(X_class, y_class)
dataloader_class = DataLoader(dataset_class, batch_size=batch_size, shuffle=True)

# Define the LSTM model for classification
class LSTMModelClassify(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, num_classes=2):
        super(LSTMModelClassify, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the last time step's output
        return out

# Training the classification model
model_classify = LSTMModelClassify()
criterion_class = nn.CrossEntropyLoss()
optimizer_class = torch.optim.Adam(model_classify.parameters(), lr=0.001)

for epoch in range(num_epochs):
    loop = tqdm(dataloader_class, leave=True)
    for inputs, targets in loop:
        inputs = inputs.unsqueeze(-1)  # Add feature dimension for LSTM
        outputs = model_classify(inputs)
        loss = criterion_class(outputs, targets)

        optimizer_class.zero_grad()
        loss.backward()
        optimizer_class.step()
                # Update tqdm progress bar
        loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
        loop.set_postfix(
            loss=loss.item()
        )
    if epoch % 10 == 0:
        print(f'[Classification] Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')

# Test a sample sequence for class prediction
test_seq_class = torch.tensor(data[-sequence_length:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
model_classify.eval()
with torch.no_grad():
    predicted_class = torch.argmax(model_classify(test_seq_class), dim=1).item()

# Display classification result
print(f'Predicted Class: {predicted_class}')
plt.plot(data[-sequence_length:], label="Test Data")
plt.axhline(y=threshold, color="r", linestyle="--", label="Threshold")
plt.axhline(y=(sum(data[-sequence_length:])/len(data[-sequence_length:])), color="b", linestyle="--", label="Actual Mean")
plt.title("Classification Test")
plt.legend()
plt.show()
