import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Load the trained model
# -----------------------------
model = joblib.load("linear_model.pkl")

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="CGPA to Package Predictor",
    page_icon="💼",
    layout="centered"
)

# -----------------------------
# Header Section
# -----------------------------
st.markdown("""
<div style="background-color:#4CAF50;padding:20px;border-radius:10px">
<h1 style="color:white;text-align:center;">CGPA to Package Predictor</h1>
<p style="color:white;text-align:center;">Enter your CGPA to predict your expected package in LPA.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar for input
# -----------------------------
st.sidebar.header("Input CGPA")
cgpa = st.sidebar.number_input(
    "Enter your CGPA (0.0 - 4.0):", 
    min_value=0.0, max_value=4.0, value=3.0, step=0.1
)

# -----------------------------
# Predict Button & Output
# -----------------------------
if st.button("Predict Package"):
    X_new = np.array([[cgpa]])  # Convert input to 2D
    predicted_package = model.predict(X_new)[0]

    # Show result in a styled container
    st.markdown(f"""
    <div style="background-color:#e0f7fa;padding:20px;border-radius:10px;margin-top:20px">
    <h2 style="text-align:center;color:#00796b;">Predicted Package: {predicted_package:.2f} LPA</h2>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Optional: Show model details
# -----------------------------
if st.checkbox("Show Model Details"):
    st.markdown(f"""
    <div style="background-color:#f1f8e9;padding:15px;border-radius:10px;margin-top:10px">
    <h3>Model Coefficients</h3>
    <p>Slope (m): {model.coef_[0]:.2f}</p>
    <p>Intercept (b): {model.intercept_:.2f}</p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div style="text-align:center;margin-top:50px;color:gray;">
<p>Developed with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)