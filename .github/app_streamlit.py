import streamlit as st
import joblib
import numpy as np

model = joblib.load('insurance_model.pkl')

st.title('Insurance Premium Calculator')

age = st.number_input('Age', min_value=18, max_value=66, value=30)
diabetes = st.selectbox('Diabetes', [0, 1])
blood_pressure = st.selectbox('Blood Pressure Problems', [0, 1])
transplants = st.selectbox('Any Transplants', [0, 1])
chronic_diseases = st.selectbox('Any Chronic Diseases', [0, 1])
height = st.number_input('Height (cm)', min_value=145, max_value=188, value=170)
weight = st.number_input('Weight (kg)', min_value=51, max_value=132, value=70)
allergies = st.selectbox('Known Allergies', [0, 1])
cancer_history = st.selectbox('History of Cancer in Family', [0, 1])
surgeries = st.number_input('Number of Major Surgeries', min_value=0, max_value=3, value=0)

features = np.array([[age, diabetes, blood_pressure, transplants, chronic_diseases,
                      height, weight, allergies, cancer_history, surgeries]])

if st.button('Predict'):
    prediction = model.predict(features)[0]
    st.write(f"Estimated Premium Price: ₹{prediction:.2f}")
