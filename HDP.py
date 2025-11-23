import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model, scaler, and feature columns
try:
    best_rf_model = joblib.load('streamlit_model/best_rf_model.joblib')
    scaler = joblib.load('streamlit_model/scaler.joblib')
    X_train_columns = joblib.load('streamlit_model/X_train_columns.joblib')
except FileNotFoundError:
    st.error("Error: Model files not found. Please ensure 'best_rf_model.joblib', 'scaler.joblib', and 'X_train_columns.joblib' are in the 'streamlit_model' directory.")
    st.stop()

# Title of the Streamlit application
st.title("Heart Disease Prediction Application")
st.write("Enter the patient's details to predict the likelihood of heart disease.")

# Input widgets for the 11 features
st.sidebar.header("Patient Information")

age = st.sidebar.slider("Age", 20, 80, 50)
sex = st.sidebar.radio("Sex", ("Male", "Female"))
chest_pain_type = st.sidebar.selectbox("Chest Pain Type", ("Typical Angina (TA)", "Atypical Angina (ATA)", "Non-Anginal Pain (NAP)", "Asymptomatic (ASY)"))
resting_bp = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.sidebar.slider("Cholesterol (mg/dl)", 0, 600, 200)
fasting_bs = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dl", ("No", "Yes"))
resting_ecg = st.sidebar.selectbox("Resting Electrocardiogram", ("Normal", "ST-T Wave Abnormality (ST)", "Left Ventricular Hypertrophy (LVH)"))
max_hr = st.sidebar.slider("Maximum Heart Rate Achieved", 60, 202, 150)
exercise_angina = st.sidebar.radio("Exercise Induced Angina", ("No", "Yes"))
oldpeak = st.sidebar.slider("Oldpeak (ST depression induced by exercise relative to rest)", 0.0, 6.2, 1.0, step=0.1)
st_slope = st.sidebar.selectbox("ST Slope", ("Upsloping (Up)", "Flat", "Downsloping (Down)"))


# Convert categorical inputs to model-compatible format
sex_map = {"Male": "M", "Female": "F"}
chest_pain_type_map = {"Typical Angina (TA)": "TA", "Atypical Angina (ATA)": "ATA", "Non-Anginal Pain (NAP)": "NAP", "Asymptomatic (ASY)": "ASY"}
fasting_bs_map = {"No": 0, "Yes": 1}
resting_ecg_map = {"Normal": "Normal", "ST-T Wave Abnormality (ST)": "ST", "Left Ventricular Hypertrophy (LVH)": "LVH"}
exercise_angina_map = {"No": "N", "Yes": "Y"}
st_slope_map = {"Upsloping (Up)": "Up", "Flat": "Flat", "Downsloping (Down)": "Down"}


# Create a DataFrame from user inputs
input_data = pd.DataFrame({
    'Age': [age],
    'Sex': [sex_map[sex]],
    'ChestPainType': [chest_pain_type_map[chest_pain_type]],
    'RestingBP': [resting_bp],
    'Cholesterol': [cholesterol],
    'FastingBS': [fasting_bs_map[fasting_bs]],
    'RestingECG': [resting_ecg_map[resting_ecg]],
    'MaxHR': [max_hr],
    'ExerciseAngina': [exercise_angina_map[exercise_angina]],
    'Oldpeak': [oldpeak],
    'ST_Slope': [st_slope_map[st_slope]]
})

# Preprocessing function
def preprocess_input(df, scaler, X_train_columns):
    # Separate numerical and categorical columns
    numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    categorical_cols = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

    # Apply One-Hot Encoding to categorical features
    df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Align columns with training data - very important for consistent predictions
    # Add missing columns with 0
    missing_cols = set(X_train_columns) - set(df_processed.columns)
    for c in missing_cols:
        df_processed[c] = 0

    # Ensure the order of columns is the same as in training data
    df_processed = df_processed[X_train_columns]

    # Scale numerical features
    df_processed[numerical_cols] = scaler.transform(df_processed[numerical_cols])

    return df_processed

# Predict button
if st.sidebar.button("Predict Heart Disease"):
    processed_input = preprocess_input(input_data, scaler, X_train_columns)
    prediction = best_rf_model.predict(processed_input)
    prediction_proba = best_rf_model.predict_proba(processed_input)[:, 1]

    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error(f"**Based on the provided information, the model predicts a HIGH likelihood of Heart Disease.**")
    else:
        st.success(f"**Based on the provided information, the model predicts a LOW likelihood of Heart Disease.**")

    st.write(f"Probability of Heart Disease: **{prediction_proba[0]:.2f}**")
    st.write("*(A probability closer to 1 indicates a higher likelihood of heart disease.)*")

st.markdown("""
---
**Note**: This application uses a machine learning model for prediction and is for informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified health provider for any questions you may have regarding a medical condition.
""")
