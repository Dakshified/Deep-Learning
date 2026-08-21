import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# Define the PyTorch LSTM Model matching Keras architecture
class StockLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim1=64, hidden_dim2=32, dense_dim=16):
        super(StockLSTM, self).__init__()
        # First LSTM layer (return_sequences=True in Keras)
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        
        # Second LSTM layer (return_sequences=False in Keras)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        
        # Fully connected Dense layers
        self.fc1 = nn.Linear(hidden_dim2, dense_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_dim, 1)
        
    def forward(self, x):
        # x shape: [batch, seq_len, features]
        # lstm1 output shape: [batch, seq_len, hidden_dim1]
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        
        # lstm2 output shape: [batch, seq_len, hidden_dim2]
        # We only take the final sequence output (equivalent to return_sequences=False)
        out, _ = self.lstm2(out)
        out = out[:, -1, :] # Final step
        out = self.dropout2(out)
        
        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def main():
    print("="*60)
    print("Training Apple (AAPL) Stock Price Predictor (PyTorch LSTM)")
    print("="*60)

    # 1. Download Stock Data
    ticker = "AAPL"
    print(f"Downloading historical stock data for {ticker}...")
    data = yf.download(ticker, period="5y", interval="1d", auto_adjust=True)
    print(f"Data downloaded! Shape: {data.shape}")
    
    close_data = data[['Close']].copy()
    if isinstance(close_data.columns, pd.MultiIndex):
        close_data.columns = close_data.columns.get_level_values(0)
    close_data.dropna(inplace=True)
    
    # Save historical plot
    plt.figure(figsize=(14, 6))
    plt.plot(close_data.index, close_data['Close'], label='Historical Closing Price')
    plt.title(f'{ticker} Historical Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Stock Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig('aapl_historical.png')
    plt.close()
    print("Saved historical price plot to aapl_historical.png")

    # 2. Normalize Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data[['Close']])
    print("Scaled Data Shape:", scaled_data.shape)

    # 3. Create Time-Series Sequences
    look_back = 60
    X = []
    y = []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    X = np.array(X)
    y = np.array(y)
    
    # Reshape to [Samples, Seq_len, Features]
    X = X.reshape(X.shape[0], X.shape[1], 1)
    print("X Shape:", X.shape)
    print("y Shape:", y.shape)

    # 4. Train-Test Split (80% train, 20% test)
    train_size = int(len(X) * 0.80)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    print(f"Training Samples: {len(X_train)} | Testing Samples: {len(X_test)}")

    # Split train into train/validation (90% train, 10% val)
    val_split = int(len(X_train) * 0.90)
    X_tr, X_val = X_train[:val_split], X_train[val_split:]
    y_tr, y_val = y_train[:val_split], y_train[val_split:]

    # Convert to PyTorch tensors
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Create PyTorch DataLoaders
    train_dataset = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 5. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StockLSTM().to(device)
    print("\nModel Architecture:")
    print(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 6. Training Loop with Early Stopping
    epochs = 50
    patience = 10
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * batch_x.size(0)
            
        epoch_train_loss = running_loss = running_train_loss / len(X_tr)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t.to(device))
            epoch_val_loss = criterion(val_outputs, y_val_t.to(device)).item()

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

        # Check Early Stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}. Restoring best weights.")
                model.load_state_dict(best_model_state)
                break

    # Save training loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('LSTM Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.savefig('aapl_training_loss.png')
    plt.close()
    print("Saved loss curve to aapl_training_loss.png")

    # 7. Predictions on Test Data
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t.to(device)).cpu().numpy()
        
    predicted_prices = scaler.inverse_transform(test_outputs).flatten()
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Calculate Metrics
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    mae = mean_absolute_error(actual_prices, predicted_prices)
    mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

    print("\nModel Performance")
    print("--------------------------")
    print("RMSE:", rmse)
    print("MAE :", mae)
    print("MAPE:", mape, "%")

    # Plot Actual vs Predicted
    test_dates = close_data.index[look_back + train_size:]
    plt.figure(figsize=(14, 6))
    plt.plot(test_dates, actual_prices, label='Actual Price')
    plt.plot(test_dates, predicted_prices, label='LSTM Predicted Price')
    plt.title(f'{ticker} - Actual vs LSTM Predicted Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Stock Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig('aapl_predictions.png')
    plt.close()
    print("Saved predictions plot to aapl_predictions.png")

    # 8. Predict Next Trading Day
    last_60_days = scaled_data[-look_back:]
    last_60_days_t = torch.tensor(last_60_days.reshape(1, look_back, 1), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        next_day_scaled = model(last_60_days_t).cpu().numpy()
        
    next_day_price = scaler.inverse_transform(next_day_scaled)[0][0]
    print(f"\nPredicted Next Trading Day Price: ${next_day_price:.2f}")

    # 9. Forecast Next One Month (21 trading days)
    future_days = 21
    future_sequence = scaled_data[-look_back:].reshape(1, look_back, 1)
    future_predictions = []

    for i in range(future_days):
        seq_tensor = torch.tensor(future_sequence, dtype=torch.float32).to(device)
        with torch.no_grad():
            pred = model(seq_tensor).cpu().numpy()
        future_predictions.append(pred[0, 0])
        
        # Shift sequence
        future_sequence = np.concatenate([future_sequence[:, 1:, :], pred.reshape(1, 1, 1)], axis=1)

    # Scale back
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

    # Generate dates
    last_date = close_data.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

    forecast_table = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': future_predictions
    })
    print("\nOne Month Forecast:")
    print(forecast_table.to_string(index=False))

    # Plot Forecast
    plt.figure(figsize=(14, 6))
    historical_dates = close_data.index[-100:]
    historical_prices = close_data['Close'].values[-100:]
    plt.plot(historical_dates, historical_prices, label='Historical Price')
    plt.plot(future_dates, future_predictions, marker='o', label='LSTM 1-Month Forecast')
    plt.axvline(x=last_date, linestyle='--', color='gray', label='Forecast Start')
    plt.title(f'{ticker} - One Month LSTM Stock Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Stock Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig('aapl_one_month_forecast.png')
    plt.close()
    print("Saved 1-month forecast plot to aapl_one_month_forecast.png")

    # Combined forecast
    plt.figure(figsize=(16, 7))
    plt.plot(close_data.index[-200:], close_data['Close'].values[-200:], label='Historical Price')
    plt.scatter(future_dates[0], future_predictions[0], s=100, color='red', label='Next Day Prediction')
    plt.plot(future_dates, future_predictions, marker='o', label='One Month Forecast')
    plt.axvline(x=last_date, linestyle='--', color='gray', label='Forecast Start')
    plt.title(f'{ticker} LSTM-Based Stock Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Stock Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig('aapl_forecast_combined.png')
    plt.close()
    print("Saved combined forecast plot to aapl_forecast_combined.png")

    # Save model weights & scaler
    torch.save(model.state_dict(), 'aapl_lstm_model.pth')
    import pickle
    with open('aapl_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Saved model weights to aapl_lstm_model.pth")
    print("Saved scaler to aapl_scaler.pkl")

    print("\n===================================")
    print("FINAL LSTM FORECAST")
    print("===================================")
    print(f"Stock: {ticker}")
    print(f"Last Actual Price: ${close_data['Close'].iloc[-1]:.2f}")
    print(f"Next Trading Day Prediction: ${next_day_price:.2f}")
    print(f"Predicted Price After {future_days} Trading Days: ${future_predictions[-1]:.2f}")
    monthly_change = ((future_predictions[-1] / close_data['Close'].iloc[-1]) - 1) * 100
    print(f"Forecasted Monthly Change: {monthly_change:.2f}%")

if __name__ == '__main__':
    main()
