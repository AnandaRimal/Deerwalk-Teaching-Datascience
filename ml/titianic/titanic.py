import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Trained Model
# -----------------------------
model = joblib.load("titanic_decision_tree.pkl")

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details below:")

# -----------------------------
# User Inputs
# -----------------------------
Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Age = st.slider("Age", 0, 80, 25)
Fare = st.number_input("Fare", min_value=0.0, value=32.0)
SibSp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
Parch = st.number_input("Parents/Children Aboard", 0, 10, 0)

Sex = st.selectbox("Sex", ["male", "female"])
Embarked = st.selectbox("Embarked Port", ["C", "Q", "S"])
Title = st.selectbox("Title", ["Mr", "Miss", "Mrs", "Rare"])

# -----------------------------
# Feature Engineering (Same as Training)
# -----------------------------
FamilySize = SibSp + Parch + 1
IsAlone = 1 if FamilySize == 1 else 0

# Get training feature names
feature_names = model.feature_names_in_

# Create empty dataframe with correct columns
input_data = pd.DataFrame(columns=feature_names)
input_data.loc[0] = 0

# Fill numeric features
input_data.at[0, "Pclass"] = Pclass
input_data.at[0, "Age"] = Age
input_data.at[0, "SibSp"] = SibSp
input_data.at[0, "Parch"] = Parch
input_data.at[0, "Fare"] = Fare
input_data.at[0, "FamilySize"] = FamilySize
input_data.at[0, "IsAlone"] = IsAlone

# -----------------------------
# Handle One-Hot Encoded Columns (drop_first=True)
# -----------------------------

# Sex (drop_first=True → only Sex_male exists)
if "Sex_male" in feature_names:
    input_data.at[0, "Sex_male"] = 1 if Sex == "male" else 0

# Embarked (drop_first=True → likely Embarked_Q and Embarked_S)
if "Embarked_Q" in feature_names:
    input_data.at[0, "Embarked_Q"] = 1 if Embarked == "Q" else 0

if "Embarked_S" in feature_names:
    input_data.at[0, "Embarked_S"] = 1 if Embarked == "S" else 0

# Title columns (drop_first=True → depends on dataset)
title_column = f"Title_{Title}"
if title_column in feature_names:
    input_data.at[0, title_column] = 1

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.success(f"🎉 Survived (Confidence: {probability[1]*100:.2f}%)")
    else:
        st.error(f"💀 Not Survived (Confidence: {probability[0]*100:.2f}%)")