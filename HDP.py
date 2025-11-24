import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model, scaler, and feature columns
best_rf_model = joblib.load('best_rf_model.joblib')
scaler = joblib.load('scaler.joblib')
X_train_columns = joblib.load('X_train_columns.joblib')

# Title of the Streamlit application
st.set_page_config(page_title="Heart Disease Prediction App", layout='wide')
st.markdown(
    "<h1 style='text-align: center;'>Heart Disease Prediction</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h3 style='text-align: center;'>Developed by Ananth Sundarrajan</h3>",
    unsafe_allow_html=True
)

st.markdown(
    "<h3 style='text-align: center;'></h3>",
    unsafe_allow_html=True
)

st.markdown("### Enter your details on the left and click 'Predict Heart Disease' to predict the likelihood of heart disease.")
st.markdown("""
**Pro-tip:** You can manually resize the sidebar by hovering your mouse cursor over the right edge of the sidebar until the cursor changes to a double-headed arrow, and then dragging the edge.
""")
# Input widgets for the 11 features
st.sidebar.header("**Enter your Information**")

age = st.sidebar.slider("**What is your Age (in years)?**", 20, 80, 50)
sex = st.sidebar.radio("**Select your gender?**", ("Male", "Female"))
chest_pain_type = st.sidebar.selectbox("**What type of 'Chest Pain' do you have?**", ("Typical Angina (TA)", "Atypical Angina (ATA)", "Non-Anginal Pain (NAP)", "Asymptomatic (ASY) or No Chest Pain"))
resting_bp = st.sidebar.slider("**What is your 'Resting Systolic Blood Pressure' in mmHg?**", 80, 200, 120)
cholesterol = st.sidebar.slider("**What is your cholesterol level in mg/dl**", 0, 600, 200)
fasting_bs = st.sidebar.radio("**Was your 'Fasting Blood Sugar' GREATER THAN 120 mg/dl?**", ("No", "Yes"))
resting_ecg = st.sidebar.selectbox("What was your 'Resting Electrocardiogram' result?", ("Normal", "ST-T Wave Abnormality (ST)", "Left Ventricular Hypertrophy (LVH)"))
max_hr = st.sidebar.slider("**What is the 'Maximum Heart Rate' achieved?**", 60, 202, 150)
exercise_angina = st.sidebar.radio("Do you have 'Exercise Induced Angina'?", ("No", "Yes"))
st.sidebar.markdown("**What is your Oldpeak score?**  \nThis is the ST depression induced by exercise relative to rest which can be found on your ECG")

oldpeak = st.sidebar.slider(
    "Use the slider to select your score:",
    0.0,
    6.2,
    1.0,
    step=0.1
)
#oldpeak = st.sidebar.slider("What is your Oldpeak score?**  \nThis is the ST depression induced by exercise relative to rest which can be found on your ECG", 0.0, 6.2, 1.0, step=0.1)
st_slope = st.sidebar.selectbox("**What was the slot of ST on your ECG?**", ("Upsloping (Up)", "Flat", "Downsloping (Down)"))


# Convert categorical inputs to model-compatible format
sex_map = {"Male": "M", "Female": "F"}
chest_pain_type_map = {"Typical Angina (TA)": "TA", "Atypical Angina (ATA)": "ATA", "Non-Anginal Pain (NAP)": "NAP", "Asymptomatic (ASY) or No Chest Pain": "ASY"}
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
