# filename: app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load saved K-Means model and scaler
kmeans = joblib.load("kmeans_customer_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Customer Segmentation with K-Means")

st.write("""
Enter customer details to predict their segment cluster.
""")

# Input form
age = st.number_input("Age", min_value=18, max_value=100, value=25)
income = st.number_input("Annual Income ($)", min_value=1000, max_value=200000, value=40000)
spending_score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)
gender = st.selectbox("Gender", ["Male", "Female"])

# Predict button
if st.button("Predict Cluster"):
    # Prepare data
    new_data = pd.DataFrame({
        "Age": [age],
        "Income": [income],
        "Spending_Score": [spending_score],
        "Gender": [gender]
    })

    # One-Hot Encode Gender
    new_data = pd.get_dummies(new_data, columns=["Gender"])

    # Ensure all columns match training data
    # Add missing columns if necessary
    for col in scaler.feature_names_in_:
        if col not in new_data.columns:
            new_data[col] = 0

    # Scale
    scaled_new_data = scaler.transform(new_data)

    # Predict
    cluster = kmeans.predict(scaled_new_data)
    st.success(f"The customer belongs to Cluster: {cluster[0]}")