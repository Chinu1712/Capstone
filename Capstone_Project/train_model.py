import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

# Download stock price data
data = yf.download('AAPL', start='2010-01-01', end='2023-01-01')
closing_prices = data['Close'].values

# Normalize data
closing_prices = closing_prices.reshape(-1, 1)

# Prepare sequences
X = []
y = []
for i in range(60, len(closing_prices)):
    X.append(closing_prices[i-60:i, 0])
    y.append(closing_prices[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(1))

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=50, batch_size=32)

# Save the model
model.save('stock_price_lstm_model.h5')
print("Model trained and saved successfully!")
