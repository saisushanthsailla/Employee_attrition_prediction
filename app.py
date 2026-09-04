
import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👨‍💼",
    layout="wide"
)


# =========================================================
# CUSTOM GREY THEME
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #1E1E1E;
    }

    .main-title {
        text-align: center;
        color: #FFFFFF;
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #B0B0B0;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .section-title {
        color: #FFFFFF;
        font-size: 22px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    div.stButton > button {
        width: 100%;
        background-color: #808080;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
        height: 50px;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #A0A0A0;
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("employee_attrition_model.pkl")


model = load_model()


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">Employee Attrition Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predict whether an employee is likely to Leave or Stay</div>',
    unsafe_allow_html=True
)


# =========================================================
# EMPLOYEE INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">Employee Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    years_at_company = st.number_input(
        "Years at Company",
        min_value=0,
        max_value=50,
        value=5,
        step=1
    )

    job_role = st.selectbox(
        "Job Role",
        [
            "Education",
            "Finance",
            "Healthcare",
            "Media",
            "Technology"
        ]
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0.0,
        value=5000.0,
        step=500.0
    )

    number_of_promotions = st.number_input(
        "Number of Promotions",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )


with col2:

    work_life_balance = st.selectbox(
        "Work-Life Balance",
        [
            "Poor",
            "Fair",
            "Good",
            "Excellent"
        ]
    )

    job_satisfaction = st.selectbox(
        "Job Satisfaction",
        [
            "Low",
            "Medium",
            "High",
            "Very High"
        ]
    )

    overtime = st.selectbox(
        "Overtime",
        ["No", "Yes"]
    )

    distance_from_home = st.number_input(
        "Distance from Home",
        min_value=0.0,
        value=10.0,
        step=1.0
    )

    education_level = st.selectbox(
        "Education Level",
        [
            "High School",
            "Associate Degree",
            "Bachelor’s Degree",
            "Master’s Degree",
            "PhD"
        ]
    )

    marital_status = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married",
            "Divorced"
        ]
    )


with col3:

    number_of_dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )

    job_level = st.selectbox(
        "Job Level",
        [
            "Entry",
            "Mid",
            "Senior"
        ]
    )

    company_size = st.selectbox(
        "Company Size",
        [
            "Small",
            "Medium",
            "Large"
        ]
    )

    remote_work = st.selectbox(
        "Remote Work",
        ["No", "Yes"]
    )

    leadership_opportunities = st.selectbox(
        "Leadership Opportunities",
        ["No", "Yes"]
    )

    innovation_opportunities = st.selectbox(
        "Innovation Opportunities",
        ["No", "Yes"]
    )

    company_reputation = st.selectbox(
        "Company Reputation",
        [
            "Poor",
            "Fair",
            "Good",
            "Excellent"
        ]
    )

    employee_recognition = st.selectbox(
        "Employee Recognition",
        [
            "Low",
            "Medium",
            "High",
            "Very High"
        ]
    )


# =========================================================
# CREATE INPUT DATAFRAME
# =========================================================

input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Years at Company": [years_at_company],
    "Job Role": [job_role],
    "Monthly Income": [monthly_income],
    "Work-Life Balance": [work_life_balance],
    "Job Satisfaction": [job_satisfaction],
    "Number of Promotions": [number_of_promotions],
    "Overtime": [overtime],
    "Distance from Home": [distance_from_home],
    "Education Level": [education_level],
    "Marital Status": [marital_status],
    "Number of Dependents": [number_of_dependents],
    "Job Level": [job_level],
    "Company Size": [company_size],
    "Remote Work": [remote_work],
    "Leadership Opportunities": [leadership_opportunities],
    "Innovation Opportunities": [innovation_opportunities],
    "Company Reputation": [company_reputation],
    "Employee Recognition": [employee_recognition]
})


# =========================================================
# PREDICTION
# =========================================================

st.markdown("---")

predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])

with predict_col2:

    if st.button("Predict Employee Attrition"):

        prediction = model.predict(input_data)[0]

        st.markdown("---")

        st.subheader("Prediction Result")

        # -------------------------------------------------
        # Prediction Probability
        # -------------------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]

            classes = model.classes_

            probability_dict = dict(
                zip(classes, probabilities)
            )

            left_probability = probability_dict.get("Left", 0)
            stayed_probability = probability_dict.get("Stayed", 0)

        else:

            left_probability = 0
            stayed_probability = 0


        # -------------------------------------------------
        # Display Prediction
        # -------------------------------------------------

        if prediction == "Left":

            st.error(
                f"⚠️ Employee is likely to LEAVE"
            )

            if hasattr(model, "predict_proba"):

                st.write(
                    f"Probability of Leaving: "
                    f"**{left_probability * 100:.2f}%**"
                )

                st.progress(
                    int(left_probability * 100)
                )

        else:

            st.success(
                f"✅ Employee is likely to STAY"
            )

            if hasattr(model, "predict_proba"):

                st.write(
                    f"Probability of Staying: "
                    f"**{stayed_probability * 100:.2f}%**"
                )

                st.progress(
                    int(stayed_probability * 100)
                )

