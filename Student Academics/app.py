from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

st.set_page_config(page_title="Student Performance AI", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.main {background:#f6f8fc;}
.block-container {padding-top:2rem;}
h1,h2,h3 {color:#17365d;}
div[data-testid="stMetric"] {
    background:white; padding:15px; border-radius:12px;
    border:1px solid #e5eaf2;
}
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = {0: "Low", 1: "Average", 2: "High"}
CLASS_HELP = {
    "Low": "The model estimates that the student's final grade may be in the 0–9 range.",
    "Average": "The model estimates that the student's final grade may be in the 10–14 range.",
    "High": "The model estimates that the student's final grade may be in the 15–20 range."
}

CATEGORICAL_OPTIONS = {
    "school": ["GP", "MS"], "sex": ["F", "M"], "address": ["U", "R"],
    "famsize": ["GT3", "LE3"], "Pstatus": ["T", "A"],
    "Mjob": ["teacher", "health", "services", "at_home", "other"],
    "Fjob": ["teacher", "health", "services", "at_home", "other"],
    "reason": ["course", "home", "reputation", "other"],
    "guardian": ["mother", "father", "other"],
    "schoolsup": ["yes", "no"], "famsup": ["yes", "no"], "paid": ["yes", "no"],
    "activities": ["yes", "no"], "nursery": ["yes", "no"], "higher": ["yes", "no"],
    "internet": ["yes", "no"], "romantic": ["yes", "no"]
}

NUMERIC_FIELDS = {
    "age": (15, 22, 17), "Medu": (0, 4, 2), "Fedu": (0, 4, 2),
    "traveltime": (1, 4, 1), "studytime": (1, 4, 2), "failures": (0, 4, 0),
    "famrel": (1, 5, 4), "freetime": (1, 5, 3), "goout": (1, 5, 3),
    "Dalc": (1, 5, 1), "Walc": (1, 5, 1), "health": (1, 5, 3),
    "absences": (0, 93, 0), "G1": (0, 20, 10), "G2": (0, 20, 10)
}

@st.cache_resource
def load_models(mode):
    preprocessor = joblib.load(MODEL_DIR / f"{mode}_preprocessor.pkl")
    selector = joblib.load(MODEL_DIR / f"{mode}_selector.pkl")
    rf = joblib.load(MODEL_DIR / f"{mode}_random_forest.pkl")
    dnn = tf.keras.models.load_model(MODEL_DIR / f"{mode}_dnn.keras")
    return preprocessor, selector, rf, dnn

def load_metrics(mode):
    path = OUTPUT_DIR / f"{mode}_metrics.json"
    if path.exists():
        with open(path, encoding="utf-8") as file:
            return json.load(file)
    return None

def add_field(data, field, key_prefix):
    if field in CATEGORICAL_OPTIONS:
        data[field] = st.selectbox(field, CATEGORICAL_OPTIONS[field], key=f"{key_prefix}_{field}")
    else:
        low, high, default = NUMERIC_FIELDS[field]
        data[field] = st.number_input(
            field, min_value=low, max_value=high, value=default, step=1,
            key=f"{key_prefix}_{field}"
        )

st.sidebar.title("🎓 Student AI")
st.sidebar.caption("Academic Performance Prediction")
page = st.sidebar.radio("Navigation", ["Dashboard", "Predict Performance", "Model Comparison", "About Project"])

if page == "Dashboard":
    st.title("Student Academic Performance Prediction")
    st.write("A simple application using a Deep Neural Network and Random Forest.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mathematics records", "395")
    c2.metric("Original attributes", "33")
    c3.metric("Prediction classes", "3")
    st.subheader("How it works")
    st.markdown("""
    1. Enter student demographic, family, academic and social information.
    2. The application preprocesses the information using the saved training pipeline.
    3. The trained model predicts Low, Average or High academic performance.
    4. Compare the DNN prediction with the Random Forest prediction.
    """)
    st.info("This is an educational estimate, not a measure of employability or a decision about a student's future.")

elif page == "Predict Performance":
    st.title("Predict Student Performance")
    mode_label = st.radio(
        "Prediction type",
        ["Early prediction (without G1 and G2)", "Updated prediction (with G1 and G2)"]
    )
    mode = "early" if mode_label.startswith("Early") else "updated"
    if not (MODEL_DIR / f"{mode}_dnn.keras").exists():
        st.error("Trained models are missing. First run: python train_model.py")
        st.stop()

    preprocessor, selector, rf, dnn = load_models(mode)
    expected_features = list(preprocessor.feature_names_in_)
    st.caption("Fill in the details. Use the information available for the student.")
    data = {}

    with st.form("student_form"):
        st.subheader("1. Personal details")
        fields = ["school", "sex", "age", "address", "famsize", "Pstatus"]
        cols = st.columns(2)
        for i, field in enumerate(fields):
            with cols[i % 2]:
                add_field(data, field, "personal")

        st.divider()
        st.subheader("2. Family details")
        fields = ["Medu", "Fedu", "Mjob", "Fjob", "guardian", "famrel", "famsup"]
        cols = st.columns(2)
        for i, field in enumerate(fields):
            with cols[i % 2]:
                add_field(data, field, "family")

        st.divider()
        st.subheader("3. Academic details")
        fields = ["reason", "traveltime", "studytime", "failures", "schoolsup", "paid", "higher"]
        if mode == "updated":
            fields += ["G1", "G2"]
        cols = st.columns(2)
        for i, field in enumerate(fields):
            with cols[i % 2]:
                add_field(data, field, "academic")

        st.divider()
        st.subheader("4. Social and behavioural details")
        fields = ["activities", "nursery", "internet", "romantic", "freetime", "goout", "Dalc", "Walc", "health", "absences"]
        cols = st.columns(2)
        for i, field in enumerate(fields):
            with cols[i % 2]:
                add_field(data, field, "social")

        submitted = st.form_submit_button("Predict performance", use_container_width=True)

    if submitted:
        try:
            input_df = pd.DataFrame([data])[expected_features]
            prepared = preprocessor.transform(input_df)
            selected = selector.transform(prepared)
            if hasattr(selected, "toarray"):
                selected = selected.toarray()

            rf_class = int(rf.predict(selected)[0])
            rf_probs = rf.predict_proba(selected)[0]
            dnn_probs = dnn.predict(selected, verbose=0)[0]
            dnn_class = int(np.argmax(dnn_probs))

            st.divider()
            st.subheader("Prediction results")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Deep Neural Network")
                st.success(f"Predicted category: **{CLASS_NAMES[dnn_class]}**")
                st.metric("Model confidence estimate", f"{dnn_probs[dnn_class] * 100:.1f}%")
                st.caption(CLASS_HELP[CLASS_NAMES[dnn_class]])
            with col2:
                st.markdown("### Random Forest")
                st.info(f"Predicted category: **{CLASS_NAMES[rf_class]}**")
                st.metric("Model confidence estimate", f"{rf_probs[rf_class] * 100:.1f}%")
                st.caption(CLASS_HELP[CLASS_NAMES[rf_class]])

            chart_df = pd.DataFrame({
                "DNN": dnn_probs,
                "Random Forest": rf_probs
            }, index=["Low", "Average", "High"])
            st.subheader("Class probability comparison")
            st.bar_chart(chart_df)
            st.warning("Predictions are estimates based on historical school data. Do not use them to judge employability, admission, or a student's potential.")
        except Exception as error:
            st.error(f"Prediction could not be completed: {error}")

elif page == "Model Comparison":
    st.title("Model Performance Comparison")
    st.write("Metrics are calculated on a held-out test set during training.")
    mode = st.selectbox("Choose experiment", ["early", "updated"])
    metrics = load_metrics(mode)
    if metrics is None:
        st.warning("Metrics are not available. Run python train_model.py first.")
    else:
        rows = []
        for key in ["dnn", "random_forest"]:
            result = metrics[key]
            rows.append({
                "Model": result["model"],
                "Accuracy": result["accuracy"],
                "Macro Precision": result["precision_macro"],
                "Macro Recall": result["recall_macro"],
                "Macro F1-score": result["f1_macro"]
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.metric("Selected input features", metrics["selected_feature_count"])
        st.metric("DNN epochs completed", metrics["epochs_trained"])
        st.subheader("DNN classification report")
        report = pd.DataFrame(metrics["dnn"]["classification_report"]).transpose()
        st.dataframe(report.round(3), use_container_width=True)
        st.subheader("DNN confusion matrix")
        st.dataframe(pd.DataFrame(
            metrics["dnn"]["confusion_matrix"],
            index=["Actual Low", "Actual Average", "Actual High"],
            columns=["Predicted Low", "Predicted Average", "Predicted High"]
        ), use_container_width=True)

elif page == "About Project":
    st.title("About the Project")
    st.markdown("""
    **Objective:** Predict a student's final Mathematics grade category using demographic,
    family, academic, social and behavioural information.

    **Dataset:** UCI Student Performance – Mathematics course (395 records).

    **Target:** `G3` (final grade from 0 to 20).

    **Categories:**
    - Low: 0–9
    - Average: 10–14
    - High: 15–20

    **Algorithms:**
    - Deep Neural Network (TensorFlow/Keras)
    - Random Forest Classifier (traditional machine learning)

    **Preprocessing:** Median/mode imputation, standard scaling, one-hot encoding and
    mutual-information feature selection.

    **Training:** 80% training data and 20% test data, with stratified splitting.
    """)
    st.caption("Dataset source: Cortez, P. (2008), UCI Student Performance Dataset, DOI: 10.24432/C5TG7T.")
