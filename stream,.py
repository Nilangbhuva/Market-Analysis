import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Set page configuration
st.set_page_config(
    page_title="US Market Trends - Predictive Analysis",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 US Market Trends - Predictive Analysis")

@st.cache_data
def load_data():
    hourly_data = pd.read_csv(r"C:\Users\Nilang\Desktop\Market-Analysis\us_market_data_hourly_with_inflation_modified.csv", parse_dates=["observation_date"])
    return hourly_data

hourly_data = load_data()

# Sidebar configuration
st.sidebar.title("Configuration")
st.sidebar.subheader("Dataset Selection")
st.sidebar.write("The dataset is preloaded with hourly data.")

st.write("### Displaying Hourly Data")
st.dataframe(hourly_data.head(), width=1200)

# Correlation Heatmap
st.title("US Market Data Correlation Heatmap")
st.sidebar.subheader("Heatmap Settings")

columns = st.sidebar.multiselect("Select Columns for Correlation", options=hourly_data.columns.drop("observation_date"), default=hourly_data.columns.drop("observation_date"))

if len(columns) > 1:
    correlation_matrix = hourly_data[columns].corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    plt.title("Correlation Heatmap", fontsize=16)
    st.pyplot(fig)
else:
    st.warning("Please select at least two columns for the heatmap.")

# Graph Plotting
st.sidebar.subheader("Data Plotting")
if st.sidebar.checkbox("Plot Data Columns"):
    column = st.sidebar.selectbox("Select Column to Plot", hourly_data.columns[1:])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hourly_data["observation_date"], hourly_data[column], label=column, color="blue")
    ax.set_xlabel("Date")
    ax.set_ylabel(column)
    ax.set_title(f"{column} Over Time", fontsize=16)
    plt.xticks(rotation=45)
    ax.legend()
    st.pyplot(fig)

# Model Training Configuration
st.sidebar.subheader("Model Training Options")
train_size = st.sidebar.slider("Training Dataset Size (%)", 10, 90, 80)
model_choice = st.sidebar.radio("Select Model", ["Random Forest", "XGBoost", "LSTM", "Neural Network", "Support Vector Regression"])

if st.sidebar.button("Train Model"):
    st.write("### Model Training and Evaluation")
    features = hourly_data.drop(columns=["observation_date", "inflation_rate"], errors='ignore')
    target = hourly_data["inflation_rate"]
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, train_size=train_size/100, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    if model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    elif model_choice == "XGBoost":
        model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, objective='reg:squarederror', random_state=42)
    elif model_choice == "LSTM":
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
    elif model_choice == "Neural Network":
        model = Sequential()
        input_dim = X_train.shape[1]
        
        # Input Layer
        model.add(Dense(128, input_dim=input_dim, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        # Hidden Layers
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Dense(16, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        # Output Layer
        model.add(Dense(1))
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
    else:
        model = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
    
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_rmse = np.sqrt(val_mse)
    val_r2 = r2_score(y_val, y_val_pred)
    
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)
    
    st.write("### Validation Metrics")
    st.write(f"- MAE: {val_mae:.4f}")
    st.write(f"- MSE: {val_mse:.4f}")
    st.write(f"- RMSE: {val_rmse:.4f}")
    st.write(f"- R²: {val_r2:.4f}")
    
    st.write("### Test Metrics")
    st.write(f"- MAE: {test_mae:.4f}")
    st.write(f"- MSE: {test_mse:.4f}")
    st.write(f"- RMSE: {test_rmse:.4f}")
    st.write(f"- R²: {test_r2:.4f}")
    
    # Plot Actual vs Predicted
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(y_test.values, label="Actual", color="blue")
    ax.plot(y_test_pred, label="Predicted", color="red")
    ax.legend()
    plt.title(f"{model_choice}: Actual vs Predicted", fontsize=16)
    st.pyplot(fig)