import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load your traffic volume dataset
df = pd.read_csv('traffic_volume.csv')

# Preprocess your date_time column as datetime type
df['date_time'] = pd.to_datetime(df['date_time'])
df['hour'] = df['date_time'].dt.hour
# Extract the features and target variable
cols = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'traffic_volume']
df_for_training = df[cols].astype(float)

# Normalize the dataset
scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)

# Prepare your training data and labels
n_past = 14  # Number of past hours used for prediction
n_future = 1  # Number of hours to predict in the future

trainX = []
trainY = []
for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
    trainX.append(df_for_training_scaled[i - n_past:i, :-1])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, -1])

trainX, trainY = np.array(trainX), np.array(trainY)

# Define and compile your LSTM model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(n_past, len(cols) - 1), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))  # Output layer with 1 neuron for traffic volume prediction

model.compile(optimizer='adam', loss='mse')
model.summary()

# Train your model
history = model.fit(trainX, trainY, epochs=5, batch_size=16, validation_split=0.1, verbose=1)

# Make predictions
n_days_for_prediction = 15
predict_period_dates = pd.date_range(df['date_time'].iloc[-n_past], periods=n_days_for_prediction, freq='H').tolist()

# Prepare input for prediction with a length of n_past and n_features features
X_pred = trainX[-n_past:]  # Use trainX[-n_past:] instead of X_test_scaled[-n_past:]
X_pred = np.reshape(X_pred, (1, n_past, len(cols) - 1))  # Reshape to match input shape

# Make predictions
predictions = model.predict(X_pred)
predicted_traffic_volume = scaler.inverse_transform(np.concatenate((trainX[-1, -1, :-1], predictions), axis=0))

# Convert timestamps to date
forecast_dates = [date_time.strftime('%Y-%m-%d %H:%M:%S') for date_time in predict_period_dates]

# Create a dataframe with the forecasted traffic volume
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted_Traffic_Volume': predicted_traffic_volume})
forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])

# Save the forecasted traffic volume to a CSV file
forecast_df.to_csv('forecast_traffic_volume.csv', index=False)

# Make predictions on the training data
y_train_pred = model.predict(trainX)

# Inverse transform the predictions to the original scale
y_train_pred_orig = scaler.inverse_transform(np.concatenate((trainX[:, -1, :-1], y_train_pred), axis=1))[:, -1]

# Calculate metrics on the training data
train_mse = mean_squared_error(trainY_orig, y_train_pred_orig)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(trainY_orig, y_train_pred_orig)
train_r2 = r2_score(trainY_orig, y_train_pred_orig)

print("Train MSE:", train_mse)
print("Train RMSE:", train_rmse)
print("Train MAE:", train_mae)
print("Train R2:", train_r2)

# Plot the training data and predictions
plt.figure(figsize=(12, 6))
plt.plot(df['date_time'][:-len(y_test)], y_train_orig, label='Training Data')
plt.plot(df['date_time'][-len(y_test):], y_test_orig, label='Testing Data')
plt.plot(forecast_df['Date'], forecast_df['Predicted_Traffic_Volume'], label='Predictions')
plt.xlabel('Date Time')
plt.ylabel('Traffic Volume')
plt.title('Traffic Volume Forecasting using LSTM')
plt.legend()

# Save the plot as an image
plt.savefig('traffic_volume_forecast.png')

plt.show()
