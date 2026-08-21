import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, Callback
import warnings
import os

warnings.filterwarnings("ignore")

# Set Page Configuration
st.set_page_config(
    page_title="LSTM Stock Price Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Premium Aesthetics
st.markdown("""
    <style>
        /* General styling */
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1E3A8A;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #4B5563;
            margin-bottom: 2rem;
        }
        /* Metrics card container */
        .metric-card {
            background-color: #F3F4F6;
            padding: 1.2rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            border-left: 5px solid #3B82F6;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #6B7280;
            font-weight: 600;
            text-transform: uppercase;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #1F2937;
        }
        .metric-delta {
            font-size: 0.95rem;
            font-weight: 600;
        }
        .metric-delta-pos {
            color: #10B981;
        }
        .metric-delta-neg {
            color: #EF4444;
        }
    </style>
""", unsafe_allow_html=True)

# Callback to show training progress in Streamlit
class StreamlitTrainingCallback(Callback):
    def __init__(self, epochs_total, progress_bar, status_slot, loss_chart_slot):
        super().__init__()
        self.epochs_total = epochs_total
        self.progress_bar = progress_bar
        self.status_slot = status_slot
        self.loss_chart_slot = loss_chart_slot
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        
        self.losses.append(loss)
        self.val_losses.append(val_loss)
        
        # Update progress bar
        progress = (epoch + 1) / self.epochs_total
        self.progress_bar.progress(progress)
        
        # Update text
        self.status_slot.markdown(
            f"**Epoch {epoch + 1}/{self.epochs_total}** — Loss: `{loss:.6f}` | Val Loss: `{val_loss:.6f}`"
        )
        
        # Update chart
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(self.losses, label="Training Loss", color="#3B82F6", lw=2)
        ax.plot(self.val_losses, label="Validation Loss", color="#EF4444", lw=2)
        ax.set_title("Live Training Loss", fontsize=10, fontweight='bold')
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Mean Squared Error", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        self.loss_chart_slot.pyplot(fig)
        plt.close(fig)

# Application Header
st.markdown("<div class='main-header'>📈 Stock Price Forecasting Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Predict and forecast stock prices using a deep learning LSTM (Long Short-Term Memory) network.</div>", unsafe_allow_html=True)

# Initialize Session State
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'ticker' not in st.session_state:
    st.session_state.ticker = "AAPL"
if 'look_back' not in st.session_state:
    st.session_state.look_back = 60
if 'epochs' not in st.session_state:
    st.session_state.epochs = 50

# Sidebar Settings
st.sidebar.header("⚙️ Configuration Settings")

popular_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "Custom"]
if st.session_state.ticker in popular_tickers:
    default_index = popular_tickers.index(st.session_state.ticker)
else:
    default_index = popular_tickers.index("Custom")

selected_ticker = st.sidebar.selectbox("Select Stock Ticker", options=popular_tickers, index=default_index)

if selected_ticker == "Custom":
    ticker_input = st.sidebar.text_input("Enter Custom Ticker Symbol", value=st.session_state.ticker if st.session_state.ticker not in popular_tickers[:-1] else "AAPL").upper().strip()
else:
    ticker_input = selected_ticker
look_back_input = st.sidebar.number_input("Look Back Period (Days)", min_value=10, max_value=200, value=st.session_state.look_back)
epochs_input = st.sidebar.number_input("Training Epochs", min_value=1, max_value=200, value=st.session_state.epochs)
batch_size_input = st.sidebar.selectbox("Batch Size", options=[16, 32, 64, 128], index=1)
val_split_input = st.sidebar.slider("Validation Split", min_value=0.05, max_value=0.30, value=0.10, step=0.05)

# Ticker or config change clears previous training run to prevent mismatched state
if (ticker_input != st.session_state.ticker or 
    look_back_input != st.session_state.look_back or 
    epochs_input != st.session_state.epochs):
    st.session_state.ticker = ticker_input
    st.session_state.look_back = look_back_input
    st.session_state.epochs = epochs_input
    st.session_state.trained = False

# Sidebar Info Box
st.sidebar.markdown("""
---
### 📘 Steps Performed:
1. **Download Data** from Yahoo Finance.
2. **Extract Closing Price** & clean.
3. **Normalize Data** using MinMaxScaler.
4. **Create Sequences** of `look_back` days.
5. **Split Data** into Train (80%) and Test (20%).
6. **Train LSTM Model** with Dropout layers.
7. **Evaluate Model** using RMSE, MAE, MAPE.
8. **Forecast Future** business days (1 Month).
""")

# Load Stock Data
@st.cache_data
def fetch_stock_data(ticker_symbol):
    try:
        df = yf.download(ticker_symbol, period="5y", interval="1d", auto_adjust=True)
        return df
    except Exception as e:
        return None

# Fetch raw data
data = fetch_stock_data(st.session_state.ticker)

if data is None or data.empty:
    st.error(f"❌ Failed to download data for ticker '{st.session_state.ticker}'. Please check the symbol and your internet connection.")
else:
    # Prepare Close Data
    close_data = data[['Close']].copy()
    if isinstance(close_data.columns, pd.MultiIndex):
        close_data.columns = close_data.columns.get_level_values(0)
    close_data.dropna(inplace=True)

    # Display Top Overview Metrics (If trained)
    if st.session_state.trained:
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        last_price = close_data['Close'].iloc[-1]
        next_day_price = st.session_state.next_day_price
        predicted_end_price = st.session_state.future_predictions[-1]
        monthly_change = ((predicted_end_price / last_price) - 1) * 100
        
        with m_col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Last Actual Price</div>
                    <div class="metric-value">${last_price:.2f}</div>
                    <div style="font-size:0.85rem; color:#6B7280;">Date: {close_data.index[-1].strftime('%Y-%m-%d')}</div>
                </div>
            """, unsafe_allow_html=True)
            
        with m_col2:
            st.markdown(f"""
                <div class="metric-card" style="border-left-color: #10B981;">
                    <div class="metric-label">Next Day Prediction</div>
                    <div class="metric-value">${next_day_price:.2f}</div>
                    <div style="font-size:0.85rem; color:#6B7280;">Next Trading Day</div>
                </div>
            """, unsafe_allow_html=True)
            
        with m_col3:
            st.markdown(f"""
                <div class="metric-card" style="border-left-color: #8B5CF6;">
                    <div class="metric-label">1-Month Forecast Price</div>
                    <div class="metric-value">${predicted_end_price:.2f}</div>
                    <div style="font-size:0.85rem; color:#6B7280;">After 21 Trading Days</div>
                </div>
            """, unsafe_allow_html=True)
            
        with m_col4:
            delta_class = "metric-delta-pos" if monthly_change >= 0 else "metric-delta-neg"
            arrow = "▲" if monthly_change >= 0 else "▼"
            st.markdown(f"""
                <div class="metric-card" style="border-left-color: {'#10B981' if monthly_change >=0 else '#EF4444'};">
                    <div class="metric-label">Forecast Monthly Change</div>
                    <div class="metric-value {delta_class}">{arrow} {monthly_change:.2f}%</div>
                    <div style="font-size:0.85rem; color:#6B7280;">Expected Directional Trend</div>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)

    # Tabs layout
    tab_overview, tab_training, tab_testing, tab_forecast = st.tabs([
        "📊 Data Exploration", 
        "🧠 Model Training", 
        "📈 Testing Performance", 
        "🔮 Future Forecast"
    ])

    # ----------------- TAB 1: OVERVIEW -----------------
    with tab_overview:
        col_chart, col_shape = st.columns([3, 1])
        
        with col_chart:
            st.subheader(f"{st.session_state.ticker} Historical Close Price")
            fig_hist, ax_hist = plt.subplots(figsize=(10, 4.5))
            ax_hist.plot(close_data.index, close_data['Close'], color='#2563EB', lw=1.5, label='Historical Price')
            ax_hist.set_title(f"{st.session_state.ticker} Stock Prices (Past 5 Years)", fontsize=12, fontweight='bold')
            ax_hist.set_xlabel("Date")
            ax_hist.set_ylabel("Price (USD)")
            ax_hist.grid(True, linestyle="--", alpha=0.5)
            ax_hist.legend()
            st.pyplot(fig_hist)
            plt.close(fig_hist)
            
        with col_shape:
            st.subheader("Data Dimensions")
            st.info(f"**Dataset Shape:** {data.shape}")
            st.markdown(f"""
            - **Total Rows:** {data.shape[0]}
            - **Total Columns:** {data.shape[1]}
            - **Start Date:** {close_data.index[0].strftime('%Y-%m-%d')}
            - **End Date:** {close_data.index[-1].strftime('%Y-%m-%d')}
            """)
            
            st.subheader("Raw Data Preview")
            st.dataframe(close_data.head(5))

    # ----------------- TAB 2: TRAINING -----------------
    with tab_training:
        st.subheader("Train LSTM Neural Network")
        st.write("Train a stacked LSTM network with custom layers to predict stock prices.")
        
        col_train_ctrl, col_train_live = st.columns([1, 2])
        
        with col_train_ctrl:
            st.markdown("### Model Architecture Specs")
            st.code(f"""
Sequential Model:
  1. LSTM(64 units, input_shape=({st.session_state.look_back}, 1), return_seq=True)
  2. Dropout(0.2)
  3. LSTM(32 units, return_seq=False)
  4. Dropout(0.2)
  5. Dense(16 units, activation='relu')
  6. Dense(1 unit, activation='linear')
            """)
            
            train_btn = st.button("🚀 Start Model Training", type="primary")
            
        with col_train_live:
            if train_btn:
                # RUN MODEL TRAINING
                with st.spinner("Preparing data and initializing model..."):
                    # Scale data
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_data = scaler.fit_transform(close_data[['Close']])
                    
                    # Create time series
                    X = []
                    y = []
                    for i in range(st.session_state.look_back, len(scaled_data)):
                        X.append(scaled_data[i-st.session_state.look_back:i, 0])
                        y.append(scaled_data[i, 0])
                    
                    X = np.array(X)
                    y = np.array(y)
                    
                    # Reshape for LSTM
                    X = X.reshape(X.shape[0], X.shape[1], 1)
                    
                    # Split Train/Test
                    train_size = int(len(X) * 0.80)
                    X_train = X[:train_size]
                    X_test = X[train_size:]
                    y_train = y[:train_size]
                    y_test = y[train_size:]
                    
                    # Build Model
                    model = Sequential()
                    model.add(LSTM(64, return_sequences=True, input_shape=(st.session_state.look_back, 1)))
                    model.add(Dropout(0.2))
                    model.add(LSTM(32, return_sequences=False))
                    model.add(Dropout(0.2))
                    model.add(Dense(16, activation='relu'))
                    model.add(Dense(1))
                    
                    model.compile(optimizer='adam', loss='mean_squared_error')
                    
                    early_stopping = EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    )
                    
                st.write("🔄 **Training Progress:**")
                progress_bar = st.progress(0.0)
                status_slot = st.empty()
                loss_chart_slot = st.empty()
                
                # Training Callback
                streamlit_cb = StreamlitTrainingCallback(
                    st.session_state.epochs, 
                    progress_bar, 
                    status_slot, 
                    loss_chart_slot
                )
                
                # Fit Model
                history = model.fit(
                    X_train,
                    y_train,
                    epochs=st.session_state.epochs,
                    batch_size=batch_size_input,
                    validation_split=val_split_input,
                    callbacks=[early_stopping, streamlit_cb],
                    verbose=0
                )
                
                st.success("✅ Model training completed successfully!")
                
                # Predict Test Data
                predicted_scaled = model.predict(X_test)
                predicted_prices = scaler.inverse_transform(predicted_scaled).flatten()
                actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
                
                # Performance Metrics
                rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
                mae = mean_absolute_error(actual_prices, predicted_prices)
                mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
                
                # Next trading day prediction
                last_60_days = scaled_data[-st.session_state.look_back:]
                last_60_days = last_60_days.reshape(1, st.session_state.look_back, 1)
                next_day_scaled = model.predict(last_60_days)
                next_day_price = scaler.inverse_transform(next_day_scaled)[0][0]
                
                # 1 Month Future Forecast (21 trading days)
                future_days = 21
                future_sequence = scaled_data[-st.session_state.look_back:].reshape(1, st.session_state.look_back, 1)
                future_predictions = []
                for _ in range(future_days):
                    next_pred = model.predict(future_sequence, verbose=0)
                    future_predictions.append(next_pred[0, 0])
                    future_sequence = np.concatenate(
                        [future_sequence[:, 1:, :], next_pred.reshape(1, 1, 1)],
                        axis=1
                    )
                
                future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
                
                # Dates
                test_dates = close_data.index[st.session_state.look_back + train_size:]
                last_date = close_data.index[-1]
                future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
                
                # Save to session state
                st.session_state.trained = True
                st.session_state.history_loss = history.history['loss']
                st.session_state.history_val_loss = history.history['val_loss']
                st.session_state.predicted_prices = predicted_prices
                st.session_state.actual_prices = actual_prices
                st.session_state.rmse = rmse
                st.session_state.mae = mae
                st.session_state.mape = mape
                st.session_state.next_day_price = next_day_price
                st.session_state.future_predictions = future_predictions
                st.session_state.test_dates = test_dates
                st.session_state.future_dates = future_dates
                
                # Trigger a rerun to show metrics and enable other tabs
                st.rerun()
                
            elif st.session_state.trained:
                st.info("ℹ️ Model has already been trained. Modify configuration inputs or click button above to retrain.")
                
                # Display the stored final training/validation loss curve
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(st.session_state.history_loss, label='Training Loss', color='#2563EB', lw=2)
                ax.plot(st.session_state.history_val_loss, label='Validation Loss', color='#EF4444', lw=2)
                ax.set_title('LSTM Model Training & Validation Loss', fontsize=12, fontweight='bold')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Mean Squared Error')
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.warning("⚠️ Model is not trained yet. Please click the button to train the LSTM model.")

    # ----------------- TAB 3: TESTING PERFORMANCE -----------------
    with tab_testing:
        if st.session_state.trained:
            st.subheader("Model Performance Evaluation on Test Set (20% Split)")
            
            c_m1, c_m2, c_m3 = st.columns(3)
            with c_m1:
                st.metric(label="Root Mean Squared Error (RMSE)", value=f"{st.session_state.rmse:.4f}")
            with c_m2:
                st.metric(label="Mean Absolute Error (MAE)", value=f"{st.session_state.mae:.4f}")
            with c_m3:
                st.metric(label="Mean Absolute Percentage Error (MAPE)", value=f"{st.session_state.mape:.2f}%")
                
            st.markdown("---")
            
            # Plot Actual vs Predicted Stock Price
            fig_test, ax_test = plt.subplots(figsize=(12, 5.5))
            ax_test.plot(st.session_state.test_dates, st.session_state.actual_prices, label='Actual Price', color='#1E2937', lw=1.5)
            ax_test.plot(st.session_state.test_dates, st.session_state.predicted_prices, label='LSTM Predicted Price', color='#F59E0B', lw=1.5, linestyle='--')
            ax_test.set_title(f"{st.session_state.ticker} - Actual vs LSTM Predicted Stock Price", fontsize=12, fontweight='bold')
            ax_test.set_xlabel("Date")
            ax_test.set_ylabel("Stock Price (USD)")
            ax_test.grid(True, linestyle="--", alpha=0.5)
            ax_test.legend()
            st.pyplot(fig_test)
            plt.close(fig_test)
        else:
            st.warning("⚠️ Please train the model on the 'Model Training' tab first to view validation performance metrics.")

    # ----------------- TAB 4: FUTURE FORECAST -----------------
    with tab_forecast:
        if st.session_state.trained:
            st.subheader("1-Month Stock Price Forecast (Next 21 Trading Days)")
            
            col_f_table, col_f_plot = st.columns([1, 2])
            
            # Forecast Table
            with col_f_table:
                st.markdown("**Future Forecast Data Table**")
                forecast_df = pd.DataFrame({
                    'Date': st.session_state.future_dates.strftime('%Y-%m-%d'),
                    'Predicted Price (USD)': [f"${val:.2f}" for val in st.session_state.future_predictions]
                })
                st.dataframe(forecast_df, use_container_width=True, height=520)
                
            # Forecast Plots
            with col_f_plot:
                # Step 23 Plot
                st.markdown("**Step 23: Last 100 Days + 21 Days Forecast**")
                fig_f100, ax_f100 = plt.subplots(figsize=(10, 4))
                
                historical_dates = close_data.index[-100:]
                historical_prices = close_data['Close'].values[-100:]
                
                ax_f100.plot(historical_dates, historical_prices, label='Historical Price', color='#2563EB', lw=1.8)
                ax_f100.plot(st.session_state.future_dates, st.session_state.future_predictions, marker='o', color='#10B981', markersize=4, label='LSTM 1-Month Forecast')
                ax_f100.axvline(x=close_data.index[-1], color='#EF4444', linestyle='--', label='Forecast Start')
                ax_f100.set_title(f"{st.session_state.ticker} - One Month LSTM Stock Price Forecast", fontsize=11, fontweight='bold')
                ax_f100.set_xlabel("Date")
                ax_f100.set_ylabel("Stock Price (USD)")
                ax_f100.grid(True, linestyle="--", alpha=0.5)
                ax_f100.legend()
                st.pyplot(fig_f100)
                plt.close(fig_f100)
                
                # Step 24 Plot
                st.markdown("**Step 24: Combined Historical & Future Forecast (Last 200 Days)**")
                fig_f200, ax_f200 = plt.subplots(figsize=(10, 4))
                
                ax_f200.plot(close_data.index[-200:], close_data['Close'].values[-200:], label='Historical Price', color='#2563EB', lw=1.8)
                ax_f200.scatter(st.session_state.future_dates[0], st.session_state.future_predictions[0], s=100, color='#EF4444', zorder=5, label='Next Day Prediction')
                ax_f200.plot(st.session_state.future_dates, st.session_state.future_predictions, marker='o', color='#10B981', markersize=4, label='One Month Forecast')
                ax_f200.axvline(x=close_data.index[-1], color='#6B7280', linestyle='--', label='Forecast Start')
                ax_f200.set_title(f"{st.session_state.ticker} LSTM-Based Stock Price Forecast", fontsize=11, fontweight='bold')
                ax_f200.set_xlabel("Date")
                ax_f200.set_ylabel("Stock Price (USD)")
                ax_f200.grid(True, linestyle="--", alpha=0.5)
                ax_f200.legend()
                st.pyplot(fig_f200)
                plt.close(fig_f200)
                
            st.markdown("---")
            
            # Step 25 Summary Box
            st.markdown("### 📋 Final Forecast Summary")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.info(f"""
                **Stock:** {st.session_state.ticker}  
                **Last Actual Price:** ${close_data['Close'].iloc[-1]:.2f}  
                **Next Trading Day Prediction:** ${st.session_state.next_day_price:.2f}
                """)
            with col_s2:
                # Re-calculate percentage change
                last_actual = close_data['Close'].iloc[-1]
                pred_final = st.session_state.future_predictions[-1]
                monthly_change = ((pred_final / last_actual) - 1) * 100
                st.success(f"""
                **Predicted Price After {future_days} Trading Days:** ${pred_final:.2f}  
                **Forecasted Monthly Change:** {monthly_change:.2f}%
                """)
        else:
            st.warning("⚠️ Please train the model on the 'Model Training' tab first to view future forecasting.")
