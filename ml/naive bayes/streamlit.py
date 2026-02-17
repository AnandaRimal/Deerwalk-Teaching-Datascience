# weather_app.py

import streamlit as st
import pandas as pd
import joblib

# 1️⃣ Load saved model and encoder
model = joblib.load('weather_model.pkl')      # Naive Bayes model
le_weather = joblib.load('le_weather.pkl')    # Encoder for Weather

# 2️⃣ Streamlit App UI
st.title("Weather → Play Prediction")
st.write("Select the weather condition to predict whether to play or not:")

# 3️⃣ User input
weather_input = st.selectbox("Weather", ['Sunny', 'Overcast', 'Rainy'])

if st.button("Predict"):
    # 4️⃣ Encode the weather input
    input_df = pd.DataFrame({'Weather':[weather_input]})
    input_df['Weather'] = le_weather.transform(input_df['Weather'])
    
    # 5️⃣ Predict
    prediction = model.predict(input_df)[0]  # 0 or 1
    probability = model.predict_proba(input_df)[0]  # probability for No and Yes
    
    # 6️⃣ Map numeric output to text
    pred_label = "Yes" if prediction == 1 else "No"
    
    # 7️⃣ Show results
    st.success(f"Weather: {weather_input} --> Prediction: {pred_label}")
    st.info(f"Probability: No = {probability[0]*100:.1f}%, Yes = {probability[1]*100:.1f}%")
