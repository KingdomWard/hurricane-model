import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Load the CSV data
data = pd.read_csv('real_hurricane_data.csv')

# Extract the relevant columns and normalize the data
years = data['Year'].values
hurricane_counts = data['Hurricane_Count'].values

scaler = MinMaxScaler()
hurricane_counts = scaler.fit_transform(hurricane_counts.reshape(-1, 1))

# Define sequence length and create sequences and target values
sequence_length = 5
X, y = [], []

for i in range(len(hurricane_counts) - sequence_length):
    X.append(hurricane_counts[i:i + sequence_length])
    y.append(hurricane_counts[i + sequence_length])

X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build and train the RNN model
model = Sequential()
model.add(SimpleRNN(units=32, activation='tanh', input_shape=(sequence_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=100, batch_size=32)

# Make predictions for 2024 and inverse transform the scaled predictions
future_data = hurricane_counts[-sequence_length:].reshape(1, sequence_length, 1)
future_predictions = model.predict(future_data)
future_predictions = scaler.inverse_transform(future_predictions)
print("Predicted number of hurricanes in 2024:", future_predictions[0][0])

# Define test years
test_years = years[train_size + sequence_length:]

# Use Matplotlib to display the results, including a line graph for test predictions
plt.figure(figsize=(12, 6))

# Plot actual data
plt.plot(test_years, scaler.inverse_transform(y_test), label="Actual Data", marker='o')

# Line graph for predictions on the test data
y_pred = model.predict(X_test)
plt.plot(test_years, scaler.inverse_transform(y_pred), label="Test Predictions", color='red', linestyle='dashed')

# Predictions for 2024
plt.plot([2024], future_predictions[0][0], marker='o', color='green', label="2024 Prediction")

plt.legend()
plt.xlabel('Year')
plt.ylabel('Hurricane Count')
plt.show()
