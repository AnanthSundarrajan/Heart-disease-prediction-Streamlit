
import streamlit as st
import pandas as pd
import joblib

# Load the model and scaler
model = joblib.load('random_forest_model.joblib')
scaler = joblib.load('scaler.joblib')

# Streamlit app title
st.title('Heart Disease Prediction App')
st.write('Enter patient details to predict the likelihood of heart disease.')

# Input fields for patient data
age = st.slider('Age', 28, 77, 50)
sex = st.selectbox('Sex', ['M', 'F'])
chest_pain_type = st.selectbox('Chest Pain Type', ['ATA', 'NAP', 'ASY', 'TA'])
resting_bp = st.slider('Resting Blood Pressure (RestingBP)', 0, 200, 120)
cholesterol = st.slider('Cholesterol', 0, 603, 200)
fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (FastingBS)', [0, 1])
resting_ecg = st.selectbox('Resting Electrocardiographic Results (RestingECG)', ['Normal', 'ST', 'LVH'])
max_hr = st.slider('Maximum Heart Rate Achieved (MaxHR)', 60, 202, 150)
exercise_angina = st.selectbox('Exercise Induced Angina (ExerciseAngina)', ['N', 'Y'])
oldpeak = st.slider('Oldpeak: ST depression induced by exercise relative to rest', -2.6, 6.2, 1.0)

# Create a DataFrame from inputs
input_data = pd.DataFrame({
    'Age': [age],
    'Sex': [sex],
    'ChestPainType': [chest_pain_type],
    'RestingBP': [resting_bp],
    'Cholesterol': [cholesterol],
    'FastingBS': [fasting_bs],
    'RestingECG': [resting_ecg],
    'MaxHR': [max_hr],
    'ExerciseAngina': [exercise_angina],
    'Oldpeak': [oldpeak]
})

# Preprocessing (matching the training pipeline)
# One-hot encode categorical features
categorical_cols = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope'] # ST_Slope will be added as a dummy

# Recreate a dummy 'ST_Slope' column with a default value for one-hot encoding reference
# Assuming 'Flat' is a common baseline or handle based on how it was encoded during training
# In our training, 'ST_Slope' had 'Down', 'Flat', 'Up' and 'drop_first=True' means 'Down' was dropped.
# So, if a user doesn't explicitly choose 'ST_Slope', we need to decide a default to avoid errors.
# For simplicity, let's assume a 'Flat' slope for initial input (if not explicitly available in input_data).
# However, the original training data does not contain 'ST_Slope' as an input feature for prediction.
# It was part of X_train, so we need to add it to the input data for prediction
# Let's add a dummy value for it if it's not present in the original input_data
# Based on X.columns, we need 'ST_Slope_Flat' and 'ST_Slope_Up'

# For simplicity, let's assume we need to replicate the full column structure of X_train
# We will use the columns from X_train directly for consistency.

# Numerical columns that were scaled
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

# List of columns exactly as they appeared in X_train after one-hot encoding and dropping
# This order is crucial for correct prediction
expected_columns = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak', 
                    'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA', 
                    'FastingBS_1', 'RestingECG_Normal', 'RestingECG_ST', 
                    'ExerciseAngina_Y', 'ST_Slope_Flat', 'ST_Slope_Up']

# Create an empty DataFrame with all expected columns and fill it
processed_input = pd.DataFrame(0, index=[0], columns=expected_columns)

# Fill numerical values
for col in numerical_cols:
    processed_input[col] = input_data[col]

# Handle one-hot encoded categorical features
# Sex
if input_data['Sex'].iloc[0] == 'M':
    processed_input['Sex_M'] = 1

# ChestPainType
if input_data['ChestPainType'].iloc[0] == 'ATA':
    processed_input['ChestPainType_ATA'] = 1
elif input_data['ChestPainType'].iloc[0] == 'NAP':
    processed_input['ChestPainType_NAP'] = 1
elif input_data['ChestPainType'].iloc[0] == 'TA':
    processed_input['ChestPainType_TA'] = 1

# FastingBS
if input_data['FastingBS'].iloc[0] == 1:
    processed_input['FastingBS_1'] = 1

# RestingECG
if input_data['RestingECG'].iloc[0] == 'Normal':
    processed_input['RestingECG_Normal'] = 1
elif input_data['RestingECG'].iloc[0] == 'ST':
    processed_input['RestingECG_ST'] = 1

# ExerciseAngina
if input_data['ExerciseAngina'].iloc[0] == 'Y':
    processed_input['ExerciseAngina_Y'] = 1

# ST_Slope
# This feature was not taken as direct input. For the app, we need to add a way for the user to select it or use a default.
# To keep it simple, I will add an input for ST_Slope here, as it's critical for the model.
# Let's assume the user selects ST_Slope now.
st_slope = st.selectbox('ST_Slope: The slope of the peak exercise ST segment', ['Up', 'Flat', 'Down'])
if st_slope == 'Flat':
    processed_input['ST_Slope_Flat'] = 1
elif st_slope == 'Up':
    processed_input['ST_Slope_Up'] = 1


# Scale numerical features
processed_input[numerical_cols] = scaler.transform(processed_input[numerical_cols])

# Prediction button
if st.button('Predict'):
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)[:, 1]

    if prediction[0] == 1:
        st.error(f'The model predicts: **Heart Disease** (Probability: {prediction_proba[0]:.2f})')
    else:
        st.success(f'The model predicts: **No Heart Disease** (Probability: {prediction_proba[0]:.2f})')

# To run this app:
# 1. Save the code as HDP.py (as done by %%writefile)
# 2. Run `streamlit run HDP.py` in your terminal
