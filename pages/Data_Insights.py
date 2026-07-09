import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Data Insights")

df = pd.read_csv("P685_alzheimers_disease_data.csv")

st.subheader("Dataset Shape")
st.write(df.shape)

# Diagnosis Distribution
st.subheader("Diagnosis Distribution")

fig, ax = plt.subplots()
sns.countplot(x="Diagnosis", data=df, ax=ax)
st.pyplot(fig)

# Age Distribution
st.subheader("Age Distribution")

fig, ax = plt.subplots()
sns.histplot(df["Age"], kde=True, ax=ax)
st.pyplot(fig)

# MMSE Distribution
st.subheader("MMSE Score Distribution")

fig, ax = plt.subplots()
sns.histplot(df["MMSE"], kde=True, ax=ax)
st.pyplot(fig)

# Correlation with Diagnosis
st.subheader("Correlation with Diagnosis")

corr = df.corr(numeric_only=True)
target_corr = corr["Diagnosis"].sort_values()

fig, ax = plt.subplots(figsize=(8,10))
target_corr.plot(kind="barh", ax=ax)

st.pyplot(fig)