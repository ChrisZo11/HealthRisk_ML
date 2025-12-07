# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import joblib

# LOAD MODELS
dt_model = joblib.load("dt_best_model.sav")
knn_model = joblib.load("knn_best_model.sav")
scaler = joblib.load("scaler.sav")

# PAGE TITLE
st.title("Health Risk Classification")
st.markdown(
    "Predict health risk using **Decision Tree (Best Model)** and **KNN** "
    "based on lifestyle and health factors."
)

# INPUT UI
st.header("Input Health Data")

col1, col2 = st.columns(2)

with col1:
    st.text("Personal Data")
    age = st.slider("Age", 0, 100, 30)
    bmi = st.slider("BMI", 10.0, 50.0, 22.0)
    sleep = st.slider("Sleep (hours)", 0.0, 12.0, 7.0)

with col2:
    st.text("Lifestyle Habits")
    smoking = st.selectbox("Smoking", [0, 1])
    alcohol = st.selectbox("Alcohol", [0, 1])
    sugar = st.slider("Sugar Intake", 0.0, 100.0, 30.0)

# Mapping risk labels
label_map = {0: "High Risk", 1: "Low Risk"}

# PREDICTION
st.text("")
if st.button("Predict Health Risk"):
    input_data = np.array([[
        age,
        bmi,
        smoking,   
        alcohol,  
        sleep,     
        sugar
    ]])

    # Decision Tree prediction
    dt_pred = dt_model.predict(input_data)[0]

    # KNN prediction (scaled)
    input_scaled = scaler.transform(input_data)
    knn_pred = knn_model.predict(input_scaled)[0]

    # Map label ke High/Low Risk
    label_map = {0: "High Risk", 1: "Low Risk"}

    st.subheader("Prediction Result")
    st.write(f"**Decision Tree:** {label_map[dt_pred]}")
    st.write(f"**KNN:** {label_map[knn_pred]}")
