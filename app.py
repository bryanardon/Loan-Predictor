import streamlit as st
import pandas as pd
import pickle
import base64

#streamlit application that calls forest_model.sav as a loan predictor

st.set_page_config(
    page_title="Loan Predictor",
    page_icon="💰",
    layout="centered"
)

@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('test.png')

loaded_model = pickle.load(open('tree_model.sav','rb'))

st.title("💰Loan Predictor💰")
st.subheader("This is a simple loan predictor application that uses a machine learning model to predict the likelihood of getting a loan approved")
st.write("Please select the following parameters to get a prediction")

col1,col2 = st.columns(2)

with st.form('Loan Predictor'):
    with col1:
        credit_score = st.number_input(
            'Enter Credit Score',
            min_value=300,
            max_value=900,
            step=10,
            help='Enter your current Credit Score'
        )
        loan_term = st.number_input(
            'Enter Loan Term (in years)',
            min_value=1,
            max_value=30,
            step=1,
            help='Enter the loan term in years'
        )
        loan_amount = st.number_input(
            'Enter Loan Amount ($)',
            min_value=1000,
            max_value=1000000,
            step=1000,
            help='Enter the loan amount you are requesting in dollars'
        )
        income_annum = st.number_input(
            'Enter Annual Income',
            min_value=10000,
            max_value=10000000,
            step=10000,
            help='Enter your annual income in dollars'
        )
    with  col2:
        residential_assets_value = st.number_input(
            'Enter Residential Assets Value ($)',
            min_value=10000,
            max_value=10000000,
            step=10000,
            help='Single-family homes, duplexes, triplexes, condos, townhouses, vacation homes, etc.'
        )
        luxury_assets_value = st.number_input(
            'Enter Luxury Assets Value ($)',
            min_value=10000,
            max_value=10000000,
            step=10000,
            help='Luxury cars, boats, airplanes, jewelry, art, collectibles, etc.'
        )
        commercial_assets_value = st.number_input(
            'Enter Commercial Assets Value ($)',
            min_value=10000,
            max_value=10000000,
            step=10000,
            help='Commercial real estate, businesses, etc.'
        )
        no_of_dependents = st.number_input(
            'Enter Number of Dependents',
            min_value=0,
            max_value=10,
            step=1,
            help='Enter the number of dependents you have and can claim on tax forms'
        )

    submit = st.form_submit_button('Predict Approval')

if submit:
    input_data = pd.DataFrame({
        'cibil_score':[credit_score],
        'loan_term':[loan_term],
        'loan_amount':[loan_amount],
        'income_annum':[income_annum],
        'residential_assets_value':[residential_assets_value],
        'luxury_assets_value':[luxury_assets_value],
        'commercial_assets_value':[commercial_assets_value],
        'no_of_dependents':[no_of_dependents]
    })

    prediction = loaded_model.predict(input_data)
    probablity = loaded_model.predict_proba(input_data)

col3,col4 = st.columns(2)
with col3:
    if prediction == 1:
        st.success('You are eligible for a loan!')
    else:
        st.error('You are not eligible for a loan.')
with col4:
    st.write(probablity)