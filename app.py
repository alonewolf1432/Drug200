import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the trained model (make sure to save your model first)
# You would have to save your model as a .pkl file first
model = pickle.load(open('drug_model.pkl', 'rb'))

# Function to make predictions based on user input
def predict_drug(age, gender, bp, cholesterol, Na_to_K):
    # Perform prediction using the loaded model
    gender_encoded = 1 if gender == 'Female' else 0
    bp_encoded = 0 if bp == 'LOW' else (1 if bp == 'NORMAL' else 2)
    cholesterol_encoded = 1 if cholesterol == 'HIGH' else 0

    data = [[age, gender_encoded, bp_encoded, cholesterol_encoded, Na_to_K]]
    prediction = model.predict(data)
    return prediction[0]

# Streamlit UI
st.title("Drug Prediction Web App")

# Input fields for user data
age = st.number_input('Age', min_value=0, max_value=100, value=25)
gender = st.selectbox('Gender', ['Male', 'Female'])
bp = st.selectbox('Blood Pressure (BP)', ['LOW', 'NORMAL', 'HIGH'])
cholesterol = st.selectbox('Cholesterol', ['NORMAL', 'HIGH'])
Na_to_K = st.number_input('Na_to_K Ratio', min_value=0.0, max_value=50.0, value=15.0)

# Button to predict
if st.button('Predict'):
    result = predict_drug(age, gender, bp, cholesterol, Na_to_K)
    st.success(f"The predicted drug is: {result}")
