import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np

# Load model
model = joblib.load("loan_model.pkl")

# Load feature names
features = joblib.load("features.pkl")

# SHAP explainer
explainer = shap.TreeExplainer(model)

st.title("Loan Approval Prediction")

st.header("Enter Applicant Details")

try:
    age = int(st.text_input("Age", "25"))

    annual_income = float(st.text_input("Annual Income", "100000"))

    credit_score = int(st.text_input("Credit Score", "700"))

    loan_amount = float(st.text_input("Loan Amount", "50000"))

    existing_loans_count = int(st.text_input("Existing Loans", "1"))

    num_dependents = int(st.text_input("Number of Dependents", "1"))

except ValueError:

    st.error("Please enter valid numeric values.")
    
    st.stop()

employment = st.selectbox(
    "Employment Status",
    ["Employed","Self-employed","Unemployed"]
)

marital = st.selectbox(
    "Marital Status",
    ["Married","Single","Divorced"]
)

gender = st.selectbox(
    "Gender",
    ["Female","Male"]
)

if st.button("Predict"):

    data = pd.DataFrame({
        "age":[age],
        "annual_income":[annual_income],
        "credit_score":[credit_score],
        "loan_amount":[loan_amount],
        "existing_loans_count":[existing_loans_count],
        "num_dependents":[num_dependents],
        "employment_status_Self-employed":[1 if employment=="Self-employed" else 0],
        "employment_status_Unemployed":[1 if employment=="Unemployed" else 0],
        "marital_status_Single":[1 if marital=="Single" else 0],
        "marital_status_Married":[1 if marital=="Married" else 0],
        "gender_Male":[1 if gender=="Male" else 0]
    })

    # Arrange columns exactly as training data
    data = data[features]

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    if prediction==1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")

    st.write(f"Approval Probability: {probability:.2%}")

    # SHAP
    shap_values = explainer(data)

    values = shap_values.values[0]

    top3 = np.argsort(np.abs(values))[::-1][:3]

    st.subheader("Top 3 Factors")

    for i in top3:

        feature = features[i]

        value = data.iloc[0,i]

        impact = values[i]

        if impact>0:
            st.write(
                f"**{feature} = {value}** increased approval "
                f"(SHAP = {impact:.2f})"
            )

        else:
            st.write(
                f"**{feature} = {value}** decreased approval "
                f"(SHAP = {impact:.2f})"
            )