import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the trained model and scaler
model = joblib.load('random_forest_model.joblib')
scaler = joblib.load('scaler.joblib')

# 2. Define column names used during training
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
# FastingBS is treated as categorical for one-hot encoding as per notebook
categorical_cols_for_dummies = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# Define the full list of features in the exact order the model expects
# This order is crucial.
model_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak', 'FastingBS_1', 'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA', 'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y', 'ST_Slope_Flat', 'ST_Slope_Up']

# 3. Streamlit App Layout
st.set_page_config(page_title="Heart Disease Prediction App", layout='wide')
st.title("Heart Disease Prediction")
st.subheader("Developed by Ananth Sundarrajan")
st.markdown("### Enter the patient's details to predict the likelihood of heart disease.")

# Input fields for user data
st.subheader('Patient Information')

# Numerical inputs
age = st.slider('**What is your Age (in years)**', 18, 100, 50)
resting_bp = st.slider('**What is your "Resting Systolic Blood Pressure" in mmHg**', 80, 200, 120)
cholesterol = st.slider('**What is your cholesterol level in mg/dl**', 100, 600, 200)
fasting_bs = st.selectbox('**Was your "Fasting Blood Sugar" GREATER THAN 120 mg/dl**', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
max_hr = st.slider('**What is the "Maximum Heart Rate" Achieved**', 60, 220, 150)
st.markdown(
    "**What is your Oldpeak score?**  \nThis is the ST depression induced by exercise relative to rest which can be found on your ECG"
)

oldpeak = st.slider(
    'Enter score here:',
    0.0,
    6.2,
    1.0
)

# Categorical inputs
sex = st.selectbox('**Select your gender (M = Male / F= Female)**', options=['F', 'M'])

options_list_chest = ['TA', 'ATA', 'NAP', 'ASY']
st.markdown("""
**Chest Pain Type**
TA: Typical Angina
ATA: Atypical Angina
NAP: Non-Anginal Pain
ASY: Asymptomatic
""")
chest_pain_type = st.selectbox(
    'Select Type',
    options=options_list_chest)

options_list_ecg = ['Normal', 'LVH', 'ST']
st.markdown("""
**Resting ECG result**
Normal: Normal result
ST: Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
LVH: Showing probable or definite left ventricular hypertrophy by Estes criteria
""")
resting_ecg = st.selectbox(
    'Select Type',
    options=options_list_ecg)
exercise_angina = st.selectbox('Exercise Induced Angina', options=['N', 'Y'])
st_slope = st.selectbox('ST_Slope - the slope of the peak exercise ST segment \n Up: upsloping \n Flat: flat \n Down: downsloping)', options=['Down', 'Flat', 'Up'])

# 4. Prediction Logic
if st.button('Predict Heart Disease'):
    # Collect inputs into a dictionary
    input_data = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain_type,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs, # Now treated as categorical for encoding
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }

    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])

    # One-hot encode all categorical inputs, including FastingBS
    # This will create columns like 'Sex_M', 'ChestPainType_ATA', 'FastingBS_1', etc.
    processed_df = pd.get_dummies(input_df, columns=categorical_cols_for_dummies, drop_first=True)

    # Ensure all model_features are present in processed_df, fill missing with 0
    # This handles cases where a particular one-hot encoded column (e.g., ChestPainType_TA)
    # was not generated because the user didn't select that option.
    for feature in model_features:
        if feature not in processed_df.columns:
            processed_df[feature] = 0

    # Apply scaling to the numerical columns only
    # Ensure numerical_cols exist before scaling (they should from input_data)
    processed_df[numerical_cols] = scaler.transform(processed_df[numerical_cols])

    # Reorder columns to match the model's expected input order
    final_input_df = processed_df[model_features]

    # Make prediction
    prediction = model.predict(final_input_df)
    prediction_proba = model.predict_proba(final_input_df)[:, 1]

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error(f'Based on the provided information, Heart Disease is **Predicted**. (Probability: {prediction_proba[0]:.2f})')
    else:
        st.success(f'Based on the provided information, Heart Disease is **NOT Predicted**. (Probability: {prediction_proba[0]:.2f})')
