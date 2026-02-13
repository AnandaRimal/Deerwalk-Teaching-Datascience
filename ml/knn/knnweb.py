import streamlit as st
import joblib
import numpy as np

# load model
model = joblib.load("knn_student_model.pkl")

st.title("🎓 Student Pass/Fail Prediction")

# inputs
hours = st.number_input("Hours Studied", min_value=0, max_value=24, step=1)
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, step=1)

# predict button
if st.button("Predict"):
    data = np.array([[hours, attendance]])
    prediction = model.predict(data)

    if prediction == 1:
        st.success("✅ Student will PASS")
    else:
        st.error("❌ Student will FAIL")
