import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import base64
import plotly.express as px

model = joblib.load("lightgbm.pkl")
explainer = shap.TreeExplainer(model)

st.set_page_config(
    page_title="Loan Risk Predictor",
    page_icon="🏦",
    layout="wide"
)

def add_bg():

    with open(loan.jpg", "rb") as f:

        data = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>

        .stApp {{

        background-image: url("data:image/jpg;base64,{data}");

        background-size: cover;

        background-position: center;

        background-attachment: fixed;

        }}

        </style>
        """,

        unsafe_allow_html=True
    )

add_bg()

st.markdown("""
<style>

[data-testid="stSidebar"]{

background:rgba(0,0,0,0.65);

}

div[data-testid="stMetric"]{

background:rgba(255,255,255,0.08);

padding:15px;

border-radius:12px;

}

</style>
""", unsafe_allow_html=True)

st.title("🏦 AI Loan Risk Assessment")

st.caption(
    "Machine Learning powered credit risk analysis using LightGBM"
)

with st.sidebar:

    st.title("🏦 Borrower Details")

    st.header("Borrower Information")

    age = st.number_input(
        "Age",
        18,
        100,
        35
    )

    income = st.number_input(
        "Annual Income",
        value=85000.0
    )

    loan_amount = st.number_input(
        "Loan Amount",
        value=200000.0
    )

    interest_rate = st.number_input(
        "Interest Rate (%)",
        value=10.5
    )

    credit_score = st.number_input(
        "Credit Score",
        300,
        900,
        720
    )

    months_employed = st.number_input(
        "Months Employed",
        value=48
    )

    num_credit_lines = st.number_input(
        "Number of Credit Lines",
        value=5
    )

    employment_type = st.selectbox(
        "Employment Type",
        [
            "Unemployed",
            "Part-time",
            "Self-employed",
            "Full-time"
        ]
    )

    has_mortgage = st.checkbox("Has Mortgage")

    has_cosigner = st.checkbox("Has Co-Signer")

    has_dependents = st.checkbox("Has Dependents")

    predict = st.button("Predict Risk")

# ====================================================
# Prediction
# ====================================================

if predict:

    # -------------------------
    # Convert Boolean to Float
    # -------------------------

    has_mortgage = float(has_mortgage)
    has_cosigner = float(has_cosigner)
    has_dependents = float(has_dependents)

    # -------------------------
    # Age Bracket
    # -------------------------

    if age <= 29:
        age_bracket = 1.0
    elif age <= 45:
        age_bracket = 2.0
    elif age <= 60:
        age_bracket = 3.0
    else:
        age_bracket = 4.0

    # -------------------------
    # Employment Encoding
    # -------------------------

    emp_map = {
        "Unemployed": 0.0,
        "Part-time": 1.0,
        "Self-employed": 2.0,
        "Full-time": 3.0
    }

    employment_type_encoded = emp_map[employment_type]

    # -------------------------
    # Feature Engineering
    # -------------------------

    credit_capacity_ratio = (
        num_credit_lines * (2 - has_mortgage)
    ) / credit_score

    credit_age_density = (
        num_credit_lines / age
    )

    household_stability = (
        has_dependents +
        has_cosigner +
        has_mortgage
    )

    unbacked_principal_exposure = (
        loan_amount /
        (
            1 +
            has_mortgage +
            has_cosigner
        )
    )

    # -------------------------
    # Model Input
    # -------------------------

    prediction_df = pd.DataFrame([{
        "Age_Bracket": age_bracket,
        "InterestRate": interest_rate,
        "Credit_Age_Density": credit_age_density,
        "Income": income,
        "MonthsEmployed": months_employed,
        "Unbacked_Principal_Exposure": unbacked_principal_exposure,
        "LoanAmount": loan_amount,
        "Household_Stability_Sum": household_stability,
        "Credit_Capacity_Ratio": credit_capacity_ratio,
        "EmploymentType": employment_type_encoded
    }]).astype(float)

    # -------------------------
    # Prediction
    # -------------------------
    prediction = model.predict(prediction_df)[0]
    probabilities = model.predict_proba(prediction_df)[0]


    st.subheader("Why did the model make this prediction?")

    shap_values = explainer.shap_values(prediction_df)

    # -------------------------
    # Display Result
    # -------------------------

    st.divider()

    confidence = max(probabilities)

    st.header("Loan Risk Assessment")

    if prediction == 0:

        st.success("🟢 LOW RISK")

    else:

        st.error("🔴 HIGH RISK")

    st.metric(
        "Model Confidence",
        f"{confidence:.2%}"
    )

# -------------------------
# Confidence
# -------------------------

    st.subheader("Prediction Confidence")

    prob_df = pd.DataFrame({
        "Risk": ["Low Risk", "High Risk"],
        "Probability": [
            probabilities[0] * 100,
            probabilities[1] * 100
        ]
    })

    fig = px.bar(
        prob_df,
        x="Probability",
        y="Risk",
        orientation="h",
        text="Probability",
        color="Risk"
    )

    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")

    fig.update_layout(
        xaxis_title="Confidence (%)",
        yaxis_title="",
        showlegend=False,
        height=300
    )

    st.plotly_chart(fig, use_container_width=True)

    if isinstance(shap_values, list):
        shap_to_plot = shap_values[1]
    else:
        shap_to_plot = shap_values

    fig, ax = plt.subplots(figsize=(8,5))

    shap.plots.waterfall(
        shap.Explanation(
            values=shap_to_plot[0],
            base_values=explainer.expected_value,
            data=prediction_df.iloc[0],
            feature_names=prediction_df.columns
        ),
        show=False
    )

    st.pyplot(fig)

    # -------------------------
    # Lending Recommendation
    # -------------------------

    st.subheader("Recommendation")

    if prediction == 0:

        st.success(
            "✅ Loan can be approved."
        )

    else:

        st.error(
            "❌ Reject loan or request additional collateral."
        )

    # -------------------------------
    # Borrower's Info
    # -------------------------------

    st.subheader("Borrower Summary")

    c1, c2 = st.columns(2)

    with c1:

        st.metric("Income", f"₹{income:,.0f}")

        st.metric("Loan Amount", f"₹{loan_amount:,.0f}")

        st.metric("Credit Score", credit_score)

    with c2:

        st.metric("Age", age)
    
        st.metric("Employment", employment_type)

        st.metric("Months Employed", months_employed)

    # -------------------------
    # Show Model Features
    # -------------------------

    with st.expander("View Engineered Features"):
        st.dataframe(prediction_df)
