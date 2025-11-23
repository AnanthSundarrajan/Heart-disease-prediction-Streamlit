import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the trained model and scaler
model = joblib.load('random_forest_model.joblib')
scaler = joblib.load('scaler.joblib')

# 2. Define categorical and numerical column names used during training
# Corrected numerical_cols to match the columns the scaler was fitted on in the notebook
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
categorical_cols = {
    'Sex': ['F', 'M'],
    'ChestPainType': ['ASY', 'ATA', 'NAP', 'TA'],
    'RestingECG': ['LVH', 'Normal', 'ST'],
    'ExerciseAngina': ['N', 'Y'],
    'ST_Slope': ['Down', 'Flat', 'Up']
}

# Define the full list of features in the exact order the model expects
# This needs to match the columns of X_train used for model training
model_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA', 'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y', 'ST_Slope_Flat', 'ST_Slope_Up']

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
sex = st.selectbox('**Select your gender (M = Male / F= Female)**', options=categorical_cols['Sex'])

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
exercise_angina = st.selectbox('Exercise Induced Angina', options=categorical_cols['ExerciseAngina'])
st_slope = st.selectbox('ST_Slope - the slope of the peak exercise ST segment \n Up: upsloping \n Flat: flat \n Down: downsloping)', options=categorical_cols['ST_Slope'])

# 4. Prediction Logic
if st.button('Predict Heart Disease'):
    # Collect inputs into a dictionary
    input_data = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain_type,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }

    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])

    # Apply one-hot encoding to categorical features
    # Create dummy columns for all possible categorical values to ensure consistency
    # The 'drop_first=True' behavior of pd.get_dummies needs to be replicated.
    # For each categorical column, one category is chosen as the reference (dropped), and others become new columns.
    # 'Sex': 'F' is the reference (Sex_M is created if Sex is M)
    # 'ChestPainType': 'ASY' is the reference (ChestPainType_ATA, ChestPainType_NAP, ChestPainType_TA are created)
    # 'RestingECG': 'LVH' is the reference (RestingECG_Normal, RestingECG_ST are created)
    # 'ExerciseAngina': 'N' is the reference (ExerciseAngina_Y is created)
    # 'ST_Slope': 'Down' is the reference (ST_Slope_Flat, ST_Slope_Up are created)

    input_df['Sex_M'] = (input_df['Sex'] == 'M').astype(int)
    input_df['ChestPainType_ATA'] = (input_df['ChestPainType'] == 'ATA').astype(int)
    input_df['ChestPainType_NAP'] = (input_df['ChestPainType'] == 'NAP').astype(int)
    input_df['ChestPainType_TA'] = (input_df['ChestPainType'] == 'TA').astype(int)
    input_df['RestingECG_Normal'] = (input_df['RestingECG'] == 'Normal').astype(int)
    input_df['RestingECG_ST'] = (input_df['RestingECG'] == 'ST').astype(int)
    input_df['ExerciseAngina_Y'] = (input_df['ExerciseAngina'] == 'Y').astype(int)
    input_df['ST_Slope_Flat'] = (input_df['ST_Slope'] == 'Flat').astype(int)
    input_df['ST_Slope_Up'] = (input_df['ST_Slope'] == 'Up').astype(int)


    # Drop original categorical columns
    input_df = input_df.drop(columns=list(categorical_cols.keys()))

    # Ensure all model_features are present, fill missing one-hot encoded columns with 0
    # This handles cases where a categorical value was not selected by the user, and thus its one-hot column wasn't created yet.
    for feature in model_features:
        if feature not in input_df.columns:
            input_df[feature] = 0

    # Scale numerical features
    # Only scale the numerical columns that the scaler was originally fitted on
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Reorder columns to match the model's expected input order
    final_input_df = input_df[model_features]

    # Make prediction
    prediction = model.predict(final_input_df)
    prediction_proba = model.predict_proba(final_input_df)[:, 1]

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error(f'Based on the provided information, Heart Disease is **Predicted**. (Probability: {prediction_proba[0]:.2f})')
    else:
        st.success(f'Based on the provided information, Heart Disease is **NOT Predicted**. (Probability: {prediction_proba[0]:.2f})')
