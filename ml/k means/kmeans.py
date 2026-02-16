import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

st.title("Student Clustering - KMeans")

# -----------------------------
# Load saved KMeans model
# -----------------------------
kmeans = joblib.load("kmeans_student_model.pkl")

# -----------------------------
# Create the dataset manually
# -----------------------------
data = {
    'Student': ['A','B','C','D','E','F','G','H','I','J'],
    'StudyHours': [2,3,4,5,6,7,8,9,10,5],
    'Attendance': [50,55,60,65,70,75,80,85,90,60]
}
df = pd.DataFrame(data)

# Predict clusters for existing students using the saved model
X = df[['StudyHours','Attendance']].values
df['Cluster'] = kmeans.predict(X)

# -----------------------------
# Show dataset
# -----------------------------
st.subheader("Existing Student Data with Clusters")
st.dataframe(df)

# -----------------------------
# Add new student input
# -----------------------------
st.subheader("Add New Student")
study_hours = st.number_input("Study Hours", min_value=0, max_value=15, value=5)
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=70)
student_name = st.text_input("Student Name", value="New Student")

if st.button("Predict Cluster"):
    new_point = np.array([[study_hours, attendance]])
    cluster = kmeans.predict(new_point)[0]
    st.success(f"{student_name} belongs to Cluster {cluster}")

# -----------------------------
# Plot clusters
# -----------------------------
centroids = kmeans.cluster_centers_
colors = ['red','blue']  # adjust if more clusters
fig, ax = plt.subplots(figsize=(8,6))

# Existing students
for i in range(len(X)):
    ax.scatter(X[i,0], X[i,1], color=colors[df['Cluster'][i]], s=100)

# New student
ax.scatter(study_hours, attendance, color='yellow', s=200, marker='*', label='New Student')

# Centroids
ax.scatter(centroids[:,0], centroids[:,1], color='green', marker='X', s=200, label='Centroids')

ax.set_xlabel("Study Hours")
ax.set_ylabel("Attendance (%)")
ax.set_title("K-Means Clustering of Students")
ax.legend()
ax.grid(True)
st.pyplot(fig)
