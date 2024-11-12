import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
# Generate a simple sine wave as our time series data
data = np.sin(np.linspace(0, 100, 1000))
scaler = MinMaxScaler()
data = scaler.fit_transform(data.reshape(-1, 1)).reshape(-1)
sequence_length = 50

# Prepare input-output pairs for training
def create_sequences(data, sequence_length):
    sequences = []
    labels = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
        labels.append(data[i + sequence_length])
    return np.array(sequences), np.array(labels)

X, y = create_sequences(data, sequence_length)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out,_ = self.lstm2(out)
        out = self.fc(out[:, -1, :])  # Take the last time step's output
        return out
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 30
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        inputs = inputs.unsqueeze(-1)  # Add feature dimension
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 20 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')
# Prepare the last sequence to predict the next value
test_seq = torch.tensor(data[-sequence_length:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
model.eval()
with torch.no_grad():
    predicted_value = model(test_seq).item()

# Display the result
print(f'Next value prediction: {predicted_value}')
# Plot the original sine wave and predicted value
plt.plot(data, label="Original Data")
plt.axvline(x=len(data) - sequence_length, color="r", linestyle="--", label="Prediction Point")
plt.scatter(len(data), predicted_value, color="green", label="Predicted Value")
plt.legend()
plt.show()


# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from torch.utils.data import DataLoader, TensorDataset

# # Generate a simple sine wave as our time series data
# data = np.sin(np.linspace(0, 100, 1000))
# scaler = MinMaxScaler()
# data = scaler.fit_transform(data.reshape(-1, 1)).reshape(-1)
# sequence_length = 50

# # Convert the data into binary labels for classification
# def create_sequences(data, sequence_length):
#     sequences = []
#     labels = []
#     threshold = 0.5  # Threshold to classify data points
#     for i in range(len(data) - sequence_length):
#         sequences.append(data[i:i + sequence_length])
#         labels.append(1 if (sum(data[i:i + sequence_length])/len(data[i:i + sequence_length])) > threshold else 0)
#     return np.array(sequences), np.array(labels)

# X, y = create_sequences(data, sequence_length)
# X = torch.tensor(X, dtype=torch.float32)
# y = torch.tensor(y, dtype=torch.long)  # Classification labels should be long tensors
# dataset = TensorDataset(X, y)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# # Define the LSTM model for classification
# class LSTMModel(nn.Module):
#     def __init__(self, input_size=1, hidden_size=50, num_layers=2, num_classes=2):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, num_classes)
    
#     def forward(self, x):
#         out, _ = self.lstm(x)
#         out = self.fc(out[:, -1, :])  # Take the last time step's output
#         return out

# # Initialize the model, loss function, and optimizer
# model = LSTMModel()
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # Training the model
# num_epochs = 30
# for epoch in range(num_epochs):
#     for inputs, targets in dataloader:
#         inputs = inputs.unsqueeze(-1)  # Add feature dimension for LSTM
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     if epoch % 10 == 0:
#         print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')

# # Test a sample sequence for class prediction
# d = data[-sequence_length-224:-224]
# test_seq = torch.tensor(d, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
# print(d)
# model.eval()
# with torch.no_grad():
#     predicted_class = torch.argmax(model(test_seq), dim=1).item()

# # Display the result
# print(f'Predicted Class: {predicted_class}')
# print(sum(d)/len(d))
# # Plot the original sine wave with a marker at the prediction point
# plt.plot(d, label="Original Data")
# #plt.axvline(x=len(data) - sequence_length, color="r", linestyle="--", label="Prediction Point")
# plt.legend()
# plt.show()
