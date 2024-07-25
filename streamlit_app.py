import numpy as np
import pandas as pd
import streamlit as st
import joblib
import sklearn as sklearn

model = joblib.load("./models/trained_sleep_disorder_predictor.pkl")
label_encoder = joblib.load('models/label_encoder.joblib')


# Title and description
st.title('Sleep Disorder Predictor')
st.write("""
    Welcome to the Sleep Disorder Predictor. Fill in the details below to predict the likelihood of having a sleep disorder.
""")

# Input fields
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=0, help="Enter your age")
    gender = st.selectbox('Gender', ['Male', 'Female'], help="Select your gender")
    bmi = st.selectbox('BMI', ['Normal', 'Overweight', 'Obese'], help="Select your Body Mass Index (BMI) category")

with col2:
    occupation = st.selectbox('Occupation', 
                              ['Accountant', 'Doctor', 'Engineer', 'Nurse', 'Salesperson', 'Teacher', 'Others'], 
                              help="Select your occupation")
    duration = st.number_input("Sleep Duration (Hours)", min_value=0.0, step=0.1, value=8.0, format="%.1f", help="Enter your average sleep duration in hours")
    quality = st.slider('Sleep Quality', min_value=1, max_value=10, value=5, help="Rate your sleep quality from 1 (Bad) to 10 (Good)")

with col3:
    activity = st.number_input("Physical Activity (Minutes)", min_value=0, help="Enter your daily physical activity in minutes")
    stress = st.slider('Stress Level', min_value=1, max_value=10, value=5, help="Rate your stress level from 1 (Low) to 10 (High)")
    bpm = st.number_input("Heart Rate (Beats Per Minute)", min_value=0, value=70, help="Enter your heart rate in beats per minute")
    steps = st.number_input("Daily Steps", min_value=0, value=6000, help="Enter the number of steps you take daily")

# Blood pressure inputs
st.subheader("Blood Pressure")
col4, col5 = st.columns(2)

with col4:
    systolic = st.number_input("Systolic", min_value=0, value=120, help="Enter your systolic blood pressure")

with col5:
    diastolic = st.number_input("Diastolic", min_value=0, value=80, help="Enter your diastolic blood pressure")

# Make prediction
if st.button('Predict'):
    options_object = {
        "Normal": 0,
        "Overweight": 0,
        "Obese": 0,
        "Male": 0,
        "Female": 0,
        "Accountant": 0,
        "Doctor": 0,
        "Engineer": 0,
        "Lawyer": 0,
        "Nurse": 0,
        "Others": 0,
        "Salesperson": 0,
        "Teacher": 0,
    }
    
    options_object[bmi] = 1
    options_object[gender] = 1
    options_object[occupation] = 1
    
    input_features = [age, duration, quality, activity, stress, bpm, steps]
    
    for i in options_object:
        input_features.append(options_object[i])
        
    input_features.append(systolic)
    input_features.append(diastolic)
    
    prediction = model.predict([input_features])
    st.success(f'The prediction is: {label_encoder.inverse_transform((prediction))[0]}')
