import streamlit as st
import pandas as pd

#streamlit application that calls forest_model.sav as a loan predictor

st.title("Loan Predictor")
st.write("This is a simple loan predictor application that uses a machine learning model to predict the likelihood of getting a loan approved")
st.write("Please select the following parameters to get a prediction")