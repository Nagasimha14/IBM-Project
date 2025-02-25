import streamlit as st
import requests
import numpy as np

st.title('Automated Model Selection & Optimization')

input_data = st.text_input("Enter input data (comma-separated):")

if st.button('Predict'):
    data = [float(i) for i in input_data.split(',')]
    response = requests.post('http://127.0.0.1:5000/predict', json={'features': data})
    st.write(f"Prediction: {response.json()['prediction']}")

