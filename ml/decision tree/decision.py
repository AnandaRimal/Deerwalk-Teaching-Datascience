# ================================
# STEP 1: Import Libraries
# ================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# ================================
# STEP 2: Load Trained Model
# ================================
model = joblib.load("breast_cancer_model.pkl")

# ================================
# STEP 3: Set Page Config & Title
# ================================
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🩺",
    layout="wide"
)

# Title
st.title("🩺 Breast Cancer Prediction Web App")
st.markdown("""
This app predicts whether a breast tumor is **Benign** or **Malignant** based on input features.
""")

# ================================
# STEP 4: Display Breast Cancer Image
# ================================
# Add a decorative image (download an image like breast_cancer.png)
image = Image.open("breast_cancer.png")  # Make sure this image exists in the folder
st.image(image, caption="Breast Cancer Awareness", use_column_width=True)

# ================================
# STEP 5: Create Feature Input Sliders
# ================================
st.sidebar.header("Patient Features")

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
feature_names = data.feature_names

# Collect user input
user_input = {}

for feature in feature_names:
    min_val = float(np.min(data.data[:, list(feature_names).index(feature)]))
    max_val = float(np.max(data.data[:, list(feature_names).index(feature)]))
    mean_val = float(np.mean(data.data[:, list(feature_names).index(feature)]))
    
    user_input[feature] = st.sidebar.slider(
        feature,
        min_value=min_val,
        max_value=max_val,
        value=mean_val
    )

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# ================================
# STEP 6: Make Prediction
# ================================
if st.button("Predict"):
    prediction = model.predict(input_df)[0]

    if prediction == 0:
        st.error(f"⚠️ Prediction: Malignant Tumor")
    else:
        st.success(f"✅ Prediction: Benign Tumor")

# ================================
# STEP 7: Show Raw Input Data
# ================================
with st.expander("Show Input Features"):
    st.dataframe(input_df)
