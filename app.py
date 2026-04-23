import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved pipeline
model = joblib.load('airline_satisfaction_svm_pipeline.joblib')

st.title("✈️ Airline Passenger Satisfaction Predictor")
st.markdown("Enter the passenger details below to predict satisfaction level.")

# Layout for inputs
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
    age = st.slider("Age", 7, 85, 30)

with col2:
    travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
    travel_class = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])
    distance = st.number_input("Flight Distance", value=1000)

with col3:
    dep_delay = st.number_input("Departure Delay (min)", value=0)
    arr_delay = st.number_input("Arrival Delay (min)", value=0)

st.subheader("Service Ratings (0-5)")
r_col1, r_col2 = st.columns(2)

rating_cols = [
    'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
    'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
    'Inflight entertainment', 'On-board service', 'Leg room service',
    'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness'
]

ratings = {}
for i, col in enumerate(rating_cols):
    target_col = r_col1 if i < 7 else r_col2
    ratings[col] = target_col.segmented_control(col, options=[0, 1, 2, 3, 4, 5], default=3)

if st.button("Predict Satisfaction"):
    # Preprocess inputs to match training format
    input_data = {
        'Gender': 1 if gender == "Male" else 0,
        'Customer Type': 0 if customer_type == "Loyal Customer" else 1,
        'Age': age,
        'Type of Travel': 0 if travel_type == "Business travel" else 1,
        'Class': {"Eco": 0, "Eco Plus": 1, "Business": 2}[travel_class],
        'Flight Distance': distance,
        'Total_Delay': dep_delay + arr_delay
    }
    # Add ratings
    input_data.update(ratings)
    
    # Convert to DataFrame in specific order
    feature_order = [
        'Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance',
        'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
        'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
        'Inflight entertainment', 'On-board service', 'Leg room service',
        'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness',
        'Total_Delay'
    ]
    
    df_input = pd.DataFrame([input_data])[feature_order]
    
    # Prediction
    prediction = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][prediction]

    if prediction == 1:
        st.success(f"😊 Predicted: Satisfied (Confidence: {prob:.2%})")
    else:
        st.error(f"😞 Predicted: Neutral or Dissatisfied (Confidence: {prob:.2%})")
