import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Read the CSV files
train_df = pd.read_csv('Train.csv')
test_df = pd.read_csv('Test.csv')

# Drop rows with missing hourly time entries
train_df = train_df.drop_duplicates(subset='date_time', keep='first')

# Group and average the data for each timestamp
train_df = train_df.groupby('date_time').mean().reset_index()

# Convert date_time to datetime format
train_df['date_time'] = pd.to_datetime(train_df['date_time'])
test_df['date_time'] = pd.to_datetime(test_df['date_time'])

# Selected features for training (excluding 'traffic_volume')
selected_features = ['temperature', 'wind_speed', 'clouds_all', 'traffic_volume']

# Select the relevant columns from the train and test data
train_data = train_df[selected_features]
test_data = test_df[selected_features]  # Include 'traffic_volume' in the test data

# Normalize the data
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# Prepare the training data
n_timesteps = 14
X_train, y_train = [], []

for i in range(n_timesteps, len(train_data_scaled)):
    X_train.append(train_data_scaled[i - n_timesteps:i])
    y_train.append(train_data_scaled[i][-1])

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape y_train to include an extra dimension
y_train = y_train[:, np.newaxis]

# Prepare the testing data
X_test = []

for i in range(n_timesteps, len(test_data_scaled)):
    X_test.append(test_data_scaled[i - n_timesteps:i])

X_test = np.array(X_test)

# Define the LSTM model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu'))
model.add(Dropout(0.2))

# Change Dense layer to have the appropriate number of units (1, as it's a regression problem)
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.1, verbose=1)

# Make predictions
y_test_pred = model.predict(X_test)

# Perform inverse transformation to rescale back to original range
y_test_pred_inv = scaler.inverse_transform(np.concatenate((X_test[:, -1, :-1], y_test_pred), axis=1))[:, -1]

# Evaluate the model
test_mse = mean_squared_error(test_data['temperature'][n_timesteps:], y_test_pred_inv)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(test_data['temperature'][n_timesteps:], y_test_pred_inv)
test_r2 = r2_score(test_data['temperature'][n_timesteps:], y_test_pred_inv)

print("Test MSE:", test_mse)
print("Test RMSE:", test_rmse)
print("Test MAE:", test_mae)
print("Test R2:", test_r2)

# Evaluate the model on training data
y_train_pred = model.predict(X_train)
y_train_pred_inv = scaler.inverse_transform(np.concatenate((X_train[:, -1, :-1], y_train_pred), axis=1))[:, -1]

train_mse = mean_squared_error(train_data['temperature'][n_timesteps:], y_train_pred_inv)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(train_data['temperature'][n_timesteps:], y_train_pred_inv)
train_r2 = r2_score(train_data['temperature'][n_timesteps:], y_train_pred_inv)

print("Train MSE:", train_mse)
print("Train RMSE:", train_rmse)
print("Train MAE:", train_mae)
print("Train R2:", train_r2)

# Plot the predictions against the actual test data
plt.figure(figsize=(12, 6))
plt.plot(test_df['date_time'][n_timesteps:], test_data['temperature'][n_timesteps:], label='Actual Temperature')
plt.plot(test_df['date_time'][n_timesteps:], y_test_pred_inv, label='Predicted Temperature')
plt.xlabel('Date Time')
plt.ylabel('Temperature')
plt.title('Temperature Forecasting using LSTM')
plt.legend()

# Save the plot as an image
plt.savefig('temperature_forecast.png')

plt.show()
