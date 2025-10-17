

import streamlit as st
import pandas as pd
import joblib

# Load the trained model and feature columns
model = joblib.load('best_picnic_model.pkl')
weather_df = pd.read_csv('weather_cleaned.csv')
feature_cols = [col for col in weather_df.columns if col not in ['DATE', 'MONTH', 'picnic']]

st.set_page_config(page_title="Picnic Suitability Predictor", layout="centered")
st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}
.stButton>button {
    background-color: #4f8cff;
    color: white;
    border-radius: 6px;
    font-size: 1.1em;
    padding: 0.5em 2em;
}
.st-bb {
    background: #e0e7ef;
}
.stTextInput>div>input {
    background: #fff;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

st.title('🌤️ Picnic Suitability Predictor')
st.caption('A simple, elegant tool to check if the weather is good for a picnic.')




# Show Basel and key TOURS features for prediction
basel_features = [
    'BASEL_temp_mean',
    'BASEL_humidity',
    'BASEL_cloud_cover',
    'BASEL_precipitation'
]
tours_features = [
    'TOURS_precipitation',
    'TOURS_temp_max',
    'TOURS_global_radiation',
    'TOURS_humidity',
    'TOURS_temp_mean',
    'TOURS_pressure'
]
user_input = {}
st.markdown('---')
st.subheader('Enter Basel Weather Data:')
for col in basel_features:
    user_input[col] = st.number_input(
        col.replace('BASEL_', '').replace('_', ' ').title(),
        float(weather_df[col].min()),
        float(weather_df[col].max()),
        float(weather_df[col].mean())
    )
st.markdown('---')
st.subheader('Enter TOURS Weather Data:')
for col in tours_features:
    user_input[col] = st.number_input(
        col.replace('TOURS_', '').replace('_', ' ').title(),
        float(weather_df[col].min()),
        float(weather_df[col].max()),
        float(weather_df[col].mean())
    )
st.markdown('---')
if st.button('Predict'):
    # Fill in all features for the model
    input_df = pd.DataFrame([user_input])
    for col in feature_cols:
        if col not in input_df:
            input_df[col] = weather_df[col].mean()
    input_df = input_df[feature_cols]
    prediction = model.predict(input_df)[0]
    st.markdown(f"<h3 style='color:#4f8cff;'>Picnic suitability: {'Yes' if prediction else 'No'}</h3>", unsafe_allow_html=True)
