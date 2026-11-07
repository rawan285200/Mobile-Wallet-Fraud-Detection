

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import confusion_matrix

st.set_page_config(
    page_title="Mobile Wallet Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Load Model
bundle = joblib.load(
    "models/best_fraud_model.pkl"
)

model = bundle["model"]

model_name = bundle["model_name"]

saved_threshold = bundle["threshold"]

COST_FN = bundle["cost_fn"]
COST_FP = bundle["cost_fp"]

st.title("💳 Mobile Wallet Fraud Detection")

st.write(
    "AI-powered fraud detection for mobile money transactions."
)

st.info(f"Selected Model: {model_name}")

# Feature Engineering

def prepare_features(data):

    data = data.copy()

    # Feature Engineering
    data["balanceDiffOrig"] = (
        data["oldbalanceOrg"]
        - data["newbalanceOrig"]
    )

    data["balanceDiffDest"] = (
        data["newbalanceDest"]
        - data["oldbalanceDest"]
    )

    # Remove target column
    data = data.drop(
        columns=["isFraud"],
        errors="ignore"
    )

    # Ensure exact training columns and order
    expected_columns = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "balanceDiffOrig",
        "balanceDiffDest",
        "type"
    ]

    data = data[expected_columns]

    return data



# Threshold Slider
st.subheader("⚙️ Fraud Detection Threshold")

threshold = st.slider(
    "Adjust Threshold",
    min_value=0.01,
    max_value=0.99,
    value=float(saved_threshold),
    step=0.01
)

st.write(f"Current Threshold: {threshold:.2f}")

st.divider()

# Single Transaction Form
st.subheader("🔍 Analyze Single Transaction")

with st.form("transaction_form"):

    col1, col2 = st.columns(2)

    with col1:

        step = st.number_input(
            "Transaction Step",
            min_value=1,
            value=100
        )

        transaction_type = st.selectbox(
            "Transaction Type",
            ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"]
        )

        amount = st.number_input(
            "Amount",
            min_value=0.0,
            value=1000.0
        )

        oldbalanceOrg = st.number_input(
            "Old Origin Balance",
            min_value=0.0,
            value=5000.0
        )

    with col2:

        newbalanceOrig = st.number_input(
            "New Origin Balance",
            min_value=0.0,
            value=4000.0
        )

        oldbalanceDest = st.number_input(
            "Old Destination Balance",
            min_value=0.0,
            value=2000.0
        )

        newbalanceDest = st.number_input(
            "New Destination Balance",
            min_value=0.0,
            value=3000.0
        )

    submitted = st.form_submit_button(
        "Analyze Transaction"
    )

if submitted:

    transaction = pd.DataFrame([{

        "step": step,

        "type": transaction_type,

        "amount": amount,

        "oldbalanceOrg": oldbalanceOrg,

        "newbalanceOrig": newbalanceOrig,

        "oldbalanceDest": oldbalanceDest,

        "newbalanceDest": newbalanceDest

    }])

    transaction = prepare_features(transaction)

    probability = model.predict_proba(
        transaction
    )[:, 1][0]

    prediction = (
        probability >= threshold
    )

    st.subheader("Transaction Result")

    st.metric(
        "Fraud Probability",
        f"{probability:.2%}"
    )

    if prediction:

        st.error("⚠️ Potential Fraud Detected!")

    else:

        st.success("✅ Transaction Classified as Legitimate")

    st.write(
        f"Decision Threshold: {threshold:.2f}"
    )


st.divider()

# CSV Upload
st.subheader("📂 Upload Transactions CSV")

uploaded_file = st.file_uploader(
    "Upload PaySim CSV",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    if not set([
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest"
    ]).issubset(data.columns):

        st.error("CSV is missing required PaySim columns.")

    else:

        actual_labels = (
            data["isFraud"].copy()
            if "isFraud" in data.columns
            else None
        )

        features = prepare_features(data)

        probabilities = model.predict_proba(
            features
        )[:, 1]

        predictions = (
            probabilities >= threshold
        ).astype(int)

        results = data.copy()

        results["Fraud_Score"] = probabilities

        results["Predicted_Fraud"] = predictions

        st.success("Transactions Analyzed!")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Transactions",
            len(results)
        )

        col2.metric(
            "Predicted Fraud",
            int(predictions.sum())
        )

        col3.metric(
            "Predicted Legitimate",
            int((predictions == 0).sum())
        )

        st.subheader("Prediction Results")

        st.dataframe(
            results.head(100),
            use_container_width=True
        )

        # Actual Cost Evaluation
        if actual_labels is not None:

            tn, fp, fn, tp = confusion_matrix(
                actual_labels,
                predictions,
                labels=[0, 1]
            ).ravel()

            total_cost = (
                fn * COST_FN +
                fp * COST_FP
            )

            st.subheader("💰 Cost Analysis")

            c1, c2, c3 = st.columns(3)

            c1.metric("False Alarms", int(fp))

            c2.metric("Missed Fraud", int(fn))

            c3.metric("Total Cost", f"{total_cost:,.0f}")

            st.write(
                f"False Alarm Cost: {fp * COST_FP:,.0f}"
            )

            st.write(
                f"Missed Fraud Cost: {fn * COST_FN:,.0f}"
            )

        else:

            st.info(
                "Upload a CSV containing isFraud "
                "to calculate actual False Alarms, "
                "Missed Fraud and total cost."
            )

        # Download Results
        csv = results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Predictions",
            data=csv,
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )