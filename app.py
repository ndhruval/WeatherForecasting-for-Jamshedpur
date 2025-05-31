import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Load model
model = tf.keras.models.load_model('weather_cnn_lstm.h5')

# Streamlit app
st.title("🌤️ Jamshedpur Weather Forecasting (CNN-LSTM)")
st.write("Upload past 30 days of weather data to predict tomorrow's weather.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if df.shape[0] != 30 or df.shape[1] != 3:
        st.error("CSV must have 30 rows and 3 columns: Temp_Avg, Humidity, Rainfall")
    else:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data_scaled = scaler.fit_transform(df[['Temp_Avg', 'Humidity', 'Rainfall']])
        X_input = np.expand_dims(data_scaled, axis=0)  # (1, 30, 3)

        prediction_scaled = model.predict(X_input)
        prediction = scaler.inverse_transform(prediction_scaled)[0]

        st.subheader("📈 Predicted Weather for Tomorrow")
        st.write(f"🌡️ Temperature: {prediction[0]:.2f} °C")
        st.write(f"💧 Humidity: {prediction[1]:.2f} %")
        st.write(f"🌧️ Rainfall: {prediction[2]:.2f} mm")