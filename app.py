import streamlit as st
import pandas as pd
import joblib

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Alzheimer's Disease Prediction",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------------
# Load Model
# -----------------------------------
model = joblib.load("alzheimers_gb_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------------
# Header
# -----------------------------------
st.markdown("""
<h1 style='text-align:center; color:#4F46E5;'>
🧠 Alzheimer's Disease Prediction System
</h1>
<p style='text-align:center;'>
Predict the likelihood of Alzheimer's Disease using demographic,
medical, lifestyle, cognitive and symptom-related information.
</p>
""", unsafe_allow_html=True)

st.divider()

# -----------------------------------
# Sidebar
# -----------------------------------
st.sidebar.header("About")
st.sidebar.info(
    """
    This application uses a trained Gradient Boosting model
    to predict Alzheimer's Disease risk.

    Best Model:
    - Accuracy: 95.12%
    - ROC-AUC: 94.93%
    """
)

# -----------------------------------
# Input Sections
# -----------------------------------

st.subheader("👤 Demographic Information")

col1, col2, col3 = st.columns(3)

with col1:
    Age = st.number_input("Age", 60, 100, 70)

with col2:
    Gender = st.selectbox("Gender", [0, 1])

with col3:
    Ethnicity = st.selectbox("Ethnicity", [0, 1, 2, 3])

EducationLevel = st.selectbox("Education Level", [0, 1, 2, 3])

st.divider()

st.subheader("🏃 Lifestyle Information")

col1, col2, col3 = st.columns(3)

with col1:
    BMI = st.number_input("BMI", 10.0, 50.0, 25.0)

with col2:
    Smoking = st.selectbox("Smoking", [0, 1])

with col3:
    AlcoholConsumption = st.number_input(
        "Alcohol Consumption",
        0.0,
        20.0,
        5.0
    )

PhysicalActivity = st.slider(
    "Physical Activity",
    0.0,
    10.0,
    5.0
)

DietQuality = st.slider(
    "Diet Quality",
    0.0,
    10.0,
    5.0
)

SleepQuality = st.slider(
    "Sleep Quality",
    0.0,
    10.0,
    7.0
)

st.divider()

st.subheader("🏥 Medical History")

col1, col2, col3 = st.columns(3)

with col1:
    FamilyHistoryAlzheimers = st.selectbox(
        "Family History Alzheimer's",
        [0, 1]
    )

    CardiovascularDisease = st.selectbox(
        "Cardiovascular Disease",
        [0, 1]
    )

with col2:
    Diabetes = st.selectbox(
        "Diabetes",
        [0, 1]
    )

    Depression = st.selectbox(
        "Depression",
        [0, 1]
    )

with col3:
    HeadInjury = st.selectbox(
        "Head Injury",
        [0, 1]
    )

    Hypertension = st.selectbox(
        "Hypertension",
        [0, 1]
    )

st.divider()

st.subheader("Clinical Measurements")

col1, col2 = st.columns(2)

with col1:
    SystolicBP = st.number_input(
        "Systolic BP",
        80,
        200,
        120
    )

    CholesterolTotal = st.number_input(
        "Total Cholesterol",
        100.0,
        400.0,
        200.0
    )

    CholesterolLDL = st.number_input(
        "LDL Cholesterol",
        20.0,
        300.0,
        100.0
    )

with col2:
    DiastolicBP = st.number_input(
        "Diastolic BP",
        50,
        150,
        80
    )

    CholesterolHDL = st.number_input(
        "HDL Cholesterol",
        10.0,
        150.0,
        50.0
    )

    CholesterolTriglycerides = st.number_input(
        "Triglycerides",
        20.0,
        500.0,
        150.0
    )

st.divider()

st.subheader("🧠 Cognitive & Symptom Assessment")

MMSE = st.slider("MMSE Score", 0.0, 30.0, 15.0)

FunctionalAssessment = st.slider(
    "Functional Assessment",
    0.0,
    10.0,
    5.0
)

ADL = st.slider("ADL Score", 0.0, 10.0, 5.0)

col1, col2, col3 = st.columns(3)

with col1:
    MemoryComplaints = st.selectbox(
        "Memory Complaints",
        [0, 1]
    )

    BehavioralProblems = st.selectbox(
        "Behavioral Problems",
        [0, 1]
    )

with col2:
    Confusion = st.selectbox(
        "Confusion",
        [0, 1]
    )

    Disorientation = st.selectbox(
        "Disorientation",
        [0, 1]
    )

with col3:
    PersonalityChanges = st.selectbox(
        "Personality Changes",
        [0, 1]
    )

    DifficultyCompletingTasks = st.selectbox(
        "Difficulty Completing Tasks",
        [0, 1]
    )

Forgetfulness = st.selectbox(
    "Forgetfulness",
    [0, 1]
)

st.divider()

# -----------------------------------
# Prediction
# -----------------------------------

if st.button("🔍 Predict Alzheimer's Risk"):

    input_data = pd.DataFrame([[
        Age, Gender, Ethnicity, EducationLevel, BMI,
        Smoking, AlcoholConsumption, PhysicalActivity,
        DietQuality, SleepQuality,
        FamilyHistoryAlzheimers,
        CardiovascularDisease,
        Diabetes,
        Depression,
        HeadInjury,
        Hypertension,
        SystolicBP,
        DiastolicBP,
        CholesterolTotal,
        CholesterolLDL,
        CholesterolHDL,
        CholesterolTriglycerides,
        MMSE,
        FunctionalAssessment,
        MemoryComplaints,
        BehavioralProblems,
        ADL,
        Confusion,
        Disorientation,
        PersonalityChanges,
        DifficultyCompletingTasks,
        Forgetfulness
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            f"⚠️ High Risk of Alzheimer's Disease\n\n"
            f"Probability: {probability:.2%}"
        )
    else:
        st.success(
            f"✅ Low Risk of Alzheimer's Disease\n\n"
            f"Probability: {(1-probability):.2%}"
        )

    st.progress(float(probability))

    st.metric(
        label="Risk Probability",
        value=f"{probability:.2%}"
    )
