import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Load stock data
data = pd.read_csv('data/stock_prices.csv', parse_dates=['Date'], index_col='Date')
series = data['Close'].values.reshape(-1, 1)  # Use 'Close' prices for forecasting

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
series = scaler.fit_transform(series)

seq_length = 40
# Prepare sequences
sequences, labels = [], []
for i in range(len(series) - seq_length):
    sequences.append(series[i:i + seq_length])
    labels.append(series[i + seq_length])

X = np.array(sequences)
y = np.array(labels) 

X_train = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)  # Increased batch size

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):  # Increased hidden size
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 50  # Increased epochs for better learning
for epoch in range(epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'lstm_stock_model.pth')

# Future predictions
def predict(model, data, seq_length):
    model.eval()
    predictions = []
    for i in range(len(data) - seq_length):
        inputs = torch.tensor(data[i:i+seq_length], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            predictions.append(model(inputs).numpy())
    return np.array(predictions)

# Generate predictions
predicted_values = predict(model, series, seq_length)
predicted_values = scaler.inverse_transform(predicted_values.reshape(-1, 1))
# print(f'Predicted Close Price: {predicted_values[-1][0]:.4f}')

# Plot
plt.figure(figsize=(12,6))
plt.plot(data.index[seq_length:], scaler.inverse_transform(series[seq_length:]), label='Actual Prices')
plt.plot(data.index[seq_length:], predicted_values, label='Predicted Prices', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Actual vs Predicted Stock Prices')
plt.legend()
plt.show()
