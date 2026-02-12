# ==========================================
# CGPA/IQ/Internship Package Prediction App
# ==========================================

import streamlit as st
import numpy as np
import joblib

# ------------------------------------------
# Page Configuration
# ------------------------------------------
st.set_page_config(
    page_title="Internship Package Predictor",
    page_icon="💼",
    layout="wide"
)

# ------------------------------------------
# Custom CSS Styling
# ------------------------------------------
st.markdown("""
    <style>
    .main-title {
        font-size:40px;
        font-weight:700;
        color:#2E86C1;
    }
    .prediction-box {
        padding:25px;
        border-radius:12px;
        background: linear-gradient(135deg, #1f77b4, #4CAF50);
        color: white;
        font-size:24px;
        font-weight:700;
        text-align:center;
        box-shadow: 0px 8px 20px rgba(0,0,0,0.3);
        margin-top:20px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------
# Load Model
# ------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("cgpa_iq_package_model.pkl")

model = load_model()

# ------------------------------------------
# Title Section
# ------------------------------------------
st.markdown("<div class='main-title'>💼 Internship Package Predictor</div>", unsafe_allow_html=True)
st.write("Predict your expected internship package based on CGPA, IQ, and Internship Score.")

st.divider()

# ------------------------------------------
# Sidebar Inputs
# ------------------------------------------
st.sidebar.header("Enter Student Details")

cgpa = st.sidebar.slider("CGPA", 0.0, 10.0, 8.0)
iq = st.sidebar.slider("IQ Score", 80, 150, 115)
internship = st.sidebar.slider("Internship Score", 0, 100, 80)

# ------------------------------------------
# Prediction Button
# ------------------------------------------
if st.sidebar.button("Predict Package"):

    input_data = np.array([[cgpa, iq, internship]])
    prediction = model.predict(input_data)
    predicted_package = prediction[0]

    st.subheader("📊 Prediction Result")
    st.markdown(
        f"""
        <div class='prediction-box'>
            💰 Estimated Internship Package <br><br>
            {predicted_package:.2f} LPA
        </div>
        """,
        unsafe_allow_html=True
    )

# ------------------------------------------
# Feature Explanation Section
# ------------------------------------------
st.divider()
st.subheader("Feature Descriptions")

st.write("""
- **CGPA**: Academic performance of the student  
- **IQ Score**: Intelligence quotient score  
- **Internship Score**: Evaluation score from internship  
- **Package**: Predicted internship package in LPA (Lakhs per Annum)  
""")

# ------------------------------------------
# Footer
# ------------------------------------------
st.divider()
st.caption("Built with Streamlit | Multiple Linear Regression Model")