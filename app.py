import streamlit as st
import pandas as pd
import joblib

# Load files
model = joblib.load("ford_model.pkl")
scaler = joblib.load("ford_scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("🚗 Ford Car Price Predictor")

# Inputs

year = st.number_input(
    "Year",
    min_value=1990,
    max_value=2030,
    value=2018
)

mileage = st.number_input(
    "Mileage",
    min_value=0,
    value=30000
)

tax = st.number_input(
    "Tax",
    min_value=0,
    value=150
)

mpg = st.number_input(
    "MPG",
    min_value=0.0,
    value=50.0
)

engineSize = st.number_input(
    "Engine Size",
    min_value=0.5,
    value=1.5
)

# IMPORTANT:
# Replace these values with actual values
# from your dataset

model_name = st.selectbox(
    "Model",
    [
        "Fiesta",
        "Focus",
        "Puma",
        "Kuga",
        "EcoSport",
        "C-MAX",
        "Mondeo",
        "Ka+",
        "Tourneo Custom",
        "S-MAX",
        "B-MAX",
        "Edge",
        "Tourneo Connect",
        "Grand C-MAX",
        "KA",
        "Galaxy",
        "Mustang",
        "Grand Tourneo Connect",
        "Fusion",
        "Ranger",
        "Streetka",
        "Escort",
        "Transit Tourneo"
    ]
)

transmission = st.selectbox(
    "Transmission",
    [
        "Automatic",
        "Manual",
        "Semi-Auto"
    ]
)

fuelType = st.selectbox(
    "Fuel Type",
    [
        "Petrol",
        "Diesel",
        "Hybrid",
        "Electric",
        "Other"
    ]
)

transmission = st.selectbox(
    "Transmission",
    [
        "Manual",
        "Automatic",
        "Semi-Auto"
    ]
)

fuelType = st.selectbox(
    "Fuel Type",
    [
        "Petrol",
        "Diesel",
        "Hybrid"
    ]
)

if st.button("Predict Price"):

    input_df = pd.DataFrame({
        "year":[year],
        "mileage":[mileage],
        "tax":[tax],
        "mpg":[mpg],
        "engineSize":[engineSize],
        "model":[model_name],
        "transmission":[transmission],
        "fuelType":[fuelType]
    })

    input_df = pd.get_dummies(
        input_df,
        columns=["model","transmission","fuelType"]
    )

    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    cols = [
        'year',
        'mileage',
        'tax',
        'mpg',
        'engineSize'
    ]

    input_df[cols] = scaler.transform(
        input_df[cols]
    )

    prediction = model.predict(input_df)

    st.success(
        f"Estimated Price: £{prediction[0]:,.0f}"
    )
    