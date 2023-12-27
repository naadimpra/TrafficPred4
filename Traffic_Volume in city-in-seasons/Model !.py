import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Read the CSV file
df = pd.read_csv('Traffic_volume.csv')

# Feature selection based on correlation matrix
selected_features = ['temp', 'clouds_all', 'date_time', 'snow_1h', 'traffic_volume']

# Convert 'date_time' column to datetime type
df['date_time'] = pd.to_datetime(df['date_time'])

# Select only the relevant features from the DataFrame
df_selected = df[selected_features]

# Normalize the dataset
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_selected.drop(columns=['date_time']))

# Prepare data for LSTM training
def prepare_data(data, n_past, n_future):
    X, y = [], []
    for i in range(n_past, len(data) - n_future + 1):
        X.append(data[i - n_past:i, :-1])  # Use all columns except the last one (traffic_volume)
        y.append(data[i + n_future - 1, -1])  # Use only the last column (traffic_volume) as the label
    return np.array(X), np.array(y)

# Define the number of past and future time steps
n_past = 14
n_future = 1

# Prepare data for training and testing
X, y = prepare_data(scaled_data, n_past, n_future)

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Create the LSTM model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.summary()

# Fit the model
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1, verbose=1)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Perform inverse scaling on the predictions
X_train_inv = scaler.inverse_transform(X_train[:, -1, :-1].reshape(-1, 2))  # Inverse transform the input data
y_train_pred_inv = scaler.inverse_transform(y_train_pred.reshape(-1, 1))[:, 0]  # Inverse transform the predicted output

X_test_inv = scaler.inverse_transform(X_test[:, -1, :-1].reshape(-1, 2))  # Inverse transform the input data
y_test_pred_inv = scaler.inverse_transform(y_test_pred.reshape(-1, 1))[:, 0]  # Inverse transform the predicted output

# Calculate evaluation metrics
train_mse = mean_squared_error(y_train[:, -1], y_train_pred_inv)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train[:, -1], y_train_pred_inv)
train_r2 = r2_score(y_train[:, -1], y_train_pred_inv)

test_mse = mean_squared_error(y_test[:, -1], y_test_pred_inv)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test[:, -1], y_test_pred_inv)
test_r2 = r2_score(y_test[:, -1], y_test_pred_inv)

print("Train MSE:", train_mse)
print("Train RMSE:", train_rmse)
print("Train MAE:", train_mae)
print("Train R2:", train_r2)

print("Test MSE:", test_mse)
print("Test RMSE:", test_rmse)
print("Test MAE:", test_mae)
print("Test R2:", test_r2)

# Plot the training data and predictions
plt.figure(figsize=(12, 6))
plt.plot(df['date_time'][:-len(X_test)], y_train[:, -1], label='Training Data')
plt.plot(df['date_time'][-len(X_test):], y_test[:, -1], label='Testing Data')
plt.plot(df['date_time'][-len(X_test):], y_test_pred[:, -1], label='Predictions')
plt.xlabel('Date Time')
plt.ylabel('Traffic Volume')
plt.title('Traffic Volume Forecasting using LSTM')
plt.legend()

# Save the plot as an image
plt.savefig('traffic_volume_forecast.png')

plt.show()
