import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta

# --- Part 1: Generate Synthetic CSV Data ---
# Creating 2 days of 5-minute interval data for USDT/CNY
start_time = datetime(2026, 1, 1)
timestamps = [start_time + timedelta(minutes=5*i) for i in range(600)]
# Generate a random walk centered around 7.25
prices = 7.25 + np.cumsum(np.random.normal(0, 0.001, 600))

df = pd.DataFrame({'timestamp_utc': timestamps, 'close_price': prices})
df.to_csv('test_data.csv', index=False)
print("✅ Created test_data.csv")

# --- Part 2: Train a Dummy LSTM Model ---
# The helpers.py expects an input shape of (1, 12, 1)
# 12 steps = 1 hour of 5-min data
n_steps = 12

# Generate dummy training data
X_train = np.random.rand(100, n_steps, 1)
y_train = np.random.rand(100, 1)

model = Sequential([
    LSTM(16, activation='relu', input_shape=(n_steps, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=1, verbose=0)

# Save the model in the required .h5 format
model.save('my_model.h5')
print("✅ Created my_model.h5")