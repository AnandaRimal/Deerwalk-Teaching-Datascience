import streamlit as st
import numpy as np
import joblib

# ----------------------
# Load the trained model
# ----------------------
model = joblib.load("age_insurance_model.pkl")

# ----------------------
# Streamlit Web Page
# ----------------------
st.title(" Prediction")
st.write("Enter your age to predict ")

# Input
age_input = st.number_input("Enter Age:", min_value=1, max_value=90, value=25)

# Prediction button
if st.button("Predict"):
    # Reshape input for model
    age_array = np.array([[age_input]])
    
    # Prediction
    pred_class = model.predict(age_array)[0]
    pred_prob = model.predict_proba(age_array)[0][1]  # probability of class 1 (Insurance Yes)
    
    # Show results
    if pred_class == 1:
        st.success(f"✅ You are likely to have insurance!")
    else:
        st.warning(f"❌ You are unlikely to have insurance.")
    
    st.write(f"Prediction Probability: {pred_prob:.2f}")