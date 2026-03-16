import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import io

# Streamlit App Title
st.title("Sleep and Health Matrix Analysis")

# Sidebar Configuration
st.sidebar.header("Dataset Options")
use_sample = st.sidebar.checkbox("Use Sample Data", True)

# Step 1: Load Sample Dataset
if use_sample:
    sample_size = st.sidebar.slider("Number of Samples", 100, 1000, 500)
    data = pd.DataFrame({
        'Sleep_Duration': np.random.uniform(4, 10, sample_size),
        'Time_in_Bed': np.random.uniform(5, 12, sample_size),
        'Stress_Level': np.random.randint(1, 5, sample_size),
        'Heart_Rate': np.random.randint(60, 100, sample_size),
        'BMI': np.random.uniform(18, 35, sample_size),
        'Health_Status': np.random.uniform(0, 1, sample_size)  # Continuous target
    })
    data['Sleep_Efficiency'] = data['Sleep_Duration'] / data['Time_in_Bed']
    st.info("Using Sample Dataset")

# Step 2: Convert Continuous Target into Discrete Classes
bins = [0, 0.5, 1.0]  # Define bin edges
labels = [0, 1]  # Define classes (Unhealthy = 0, Healthy = 1)
data['Health_Status'] = pd.cut(data['Health_Status'], bins=bins, labels=labels, include_lowest=True)
data['Health_Status'] = data['Health_Status'].astype(int)

# Step 3: Display Dataset
st.write("### Dataset Preview")
st.dataframe(data.head())

# Download Dataset
csv_buffer = io.StringIO()
data.to_csv(csv_buffer, index=False)
st.download_button("Download Dataset", data=csv_buffer.getvalue(), file_name="sample_dataset.csv", mime="text/csv")

# Step 4: Feature Selection
features = ['Sleep_Duration', 'Time_in_Bed', 'Stress_Level', 'Heart_Rate', 'BMI', 'Sleep_Efficiency']
target = 'Health_Status'

# Standardize Features
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# Step 5: Split Data
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Results
accuracy = accuracy_score(y_test, model.predict(X_test))
st.write(f"### Model Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
st.write("### Confusion Matrix")
conf_matrix = confusion_matrix(y_test, model.predict(X_test))
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# Feature Importance
st.write("### Feature Importance")
feature_importances = pd.Series(model.feature_importances_, index=features)
fig, ax = plt.subplots()
feature_importances.sort_values(ascending=False).plot(kind='bar', color='skyblue', ax=ax)
ax.set_title("Feature Importance")
st.pyplot(fig)

# Step 7: Prediction Input Form
st.write("### Predict Health Status")
with st.form("prediction_form"):
    sleep_duration = st.number_input("Sleep Duration (hours)", min_value=4.0, max_value=10.0, value=7.0)
    time_in_bed = st.number_input("Time in Bed (hours)", min_value=5.0, max_value=12.0, value=8.0)
    stress_level = st.number_input("Stress Level (1-4)", min_value=1, max_value=4, value=2)
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=60, max_value=100, value=75)
    bmi = st.number_input("BMI", min_value=18.0, max_value=35.0, value=25.0)
    efficiency = sleep_duration / time_in_bed

    submitted = st.form_submit_button("Predict")
    if submitted:
        input_data = np.array([[sleep_duration, time_in_bed, stress_level, heart_rate, bmi, efficiency]])
        input_data = scaler.transform(input_data)  # Apply the same scaling
        prediction = model.predict(input_data)
        health_status = "Healthy" if prediction[0] == 1 else "Unhealthy"
        st.write(f"### Predicted Health Status: {health_status}")

# Step 8: Additional Plots

# 1. Distribution of Sleep Duration
st.write("### Distribution of Sleep Duration")
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(data['Sleep_Duration'], kde=True, color='teal', ax=ax)
ax.set_title("Distribution of Sleep Duration")
ax.set_xlabel("Sleep Duration (hours)")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# 2. Stress Level vs Health Status
st.write("### Stress Level vs Health Status")
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x='Health_Status', y='Stress_Level', data=data, ax=ax)
ax.set_title("Stress Level vs Health Status")
ax.set_xlabel("Health Status (0 = Unhealthy, 1 = Healthy)")
ax.set_ylabel("Stress Level")
st.pyplot(fig)


