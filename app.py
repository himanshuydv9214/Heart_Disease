import streamlit as st
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("charges_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏥 Medical Insurance Charges Predictor")

st.write("Enter details below:")

# Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=25)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0
)

smoker = st.selectbox(
    "Smoker",
    ["No", "Yes"]
)

region = st.selectbox(
    "Region",
    ["southwest", "southeast", "northwest", "northeast"]
)

if st.button("Predict Charges"):

    # Encoding

    sex = 1 if sex == "Female" else 0
    smoker = 1 if smoker == "Yes" else 0

    region_southwest = 1 if region == "southwest" else 0

    # BMI Category

    bmi_category_obese = 1 if bmi >= 30 else 0

    # Scale numerical columns

    scaled_values = scaler.transform(
        [[age, bmi, children]]
    )

    age_scaled = scaled_values[0][0]
    bmi_scaled = scaled_values[0][1]
    children_scaled = scaled_values[0][2]

    # Final feature order EXACTLY like training

    input_data = pd.DataFrame(
        [[
            age_scaled,
            sex,
            bmi_scaled,
            children_scaled,
            smoker,
            region_southwest,
            bmi_category_obese
        ]],
        columns=[
            'age',
            'sex',
            'bmi',
            'children',
            'smoker',
            'region_southwest',
            'bmi_category_Obese'
        ]
    )

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Insurance Charges: ₹{prediction[0]:,.2f}"
    )