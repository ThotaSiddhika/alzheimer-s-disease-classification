import streamlit as st
import pandas as pd

st.title("📈 Model Performance")

st.success("Best Model: Gradient Boosting")

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", "95.12%")

with col2:
    st.metric("ROC-AUC", "94.93%")

st.markdown("---")

st.subheader("Top Features")

feature_df = pd.DataFrame({
    "Feature": [
        "FunctionalAssessment",
        "ADL",
        "MMSE",
        "MemoryComplaints",
        "BehavioralProblems"
    ],
    "Importance": [
        0.39,
        0.33,
        0.27,
        0.31,
        0.22
    ]
})

st.bar_chart(
    feature_df.set_index("Feature")
)

st.markdown("---")

st.write("""
The Gradient Boosting model achieved the highest overall
performance and was selected for deployment.
""")