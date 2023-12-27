import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load the data (replace 'Traffic_Volume.csv' with your dataset path)
data = pd.read_csv('Traffic_Volume.csv')

# Convert date_time column to datetime type
data['date_time'] = pd.to_datetime(data['date_time'])

# Sort by date_time in ascending order
data = data.sort_values('date_time')

# Extract the hour from the date_time column
data['hour'] = data['date_time'].dt.hour

# Variables for training (update with the relevant columns for your dataset)
cols = ['temp', 'hour', 'clouds_all', 'snow_1h']
target_col = 'traffic_volume'

# One-hot encode categorical columns
data = pd.get_dummies(data, columns=['holiday', 'weather_main', 'weather_description'])

# New dataframe with only training data - selected columns
df_for_training = data[cols + [target_col]].astype(float)

# Normalize the dataset
scaler = MinMaxScaler()
df_for_training_scaled = scaler.fit_transform(df_for_training)

# Separate features and target
X = df_for_training_scaled[:, :-1]
y = df_for_training_scaled[:, -1]

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the data for LSTM (samples, time steps, features)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), verbose=1)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Inverse transform predictions to original scale
y_train_pred = scaler.inverse_transform(np.concatenate((X_train.reshape(-1, X_train.shape[2]), y_train_pred), axis=1))[:, -1]
y_test_pred = scaler.inverse_transform(np.concatenate((X_test.reshape(-1, X_test.shape[2]), y_test_pred), axis=1))[:, -1]

# Evaluate the model and print the results
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("Train MSE:", train_mse)
print("Train RMSE:", train_rmse)
print("Train MAE:", train_mae)
print("Train R2:", train_r2)

print("Test MSE:", test_mse)
print("Test RMSE:", test_rmse)
print("Test MAE:", test_mae)
print("Test R2:", test_r2)

# Plot the training data and predictions
plt.figure(figsize=(15, 6))  # Increase the figure width

# Training and prediction plot
plt.plot(data['date_time'][:-len(X_test)], data['traffic_volume'][:-len(X_test)], label='Training Data', color='blue')
plt.plot(data['date_time'][-len(X_test):], data['traffic_volume'][-len(X_test):], label='True Traffic Volume', color='orange')
plt.plot(data['date_time'][-len(X_test):], y_test_pred, label='Predictions', color='green')

plt.xlabel('Date')
plt.ylabel('Traffic Volume')
plt.title('Traffic Volume Forecasting using LSTM - Training and Prediction')
plt.legend()

# Save the plot as an image
plt.savefig('traffic_volume_forecast_training_prediction_lstm.png')

plt.show()

# Plot the last 2 years' data
plt.figure(figsize=(15, 6))  # Increase the figure width

# Get the start date for the last 2 years
start_date = data['date_time'].max() - pd.DateOffset(years=2)

# Filter the data for the last 2 years
last_2_years_data = data[data['date_time'] >= start_date]

# Last 2 years' data plot
plt.plot(last_2_years_data['date_time'], last_2_years_data['traffic_volume'], label='True Traffic Volume', color='blue')

plt.xlabel('Date')
plt.ylabel('Traffic Volume')
plt.title('Last 2 Years\' Traffic Volume Data')
plt.legend()

# Save the plot as an image
plt.savefig('last_2_years_traffic_volume_lstm.png')

plt.show()
