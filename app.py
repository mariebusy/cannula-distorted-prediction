import streamlit as st
import numpy as np
import pickle
# Load the pre-trained model (saved_model.pkl)
with open('saved_steps.pkl', 'rb') as file:
    model = pickle.load(file)
# Streamlit app
st.title("Prediction of Cannula Distorted")
# Data entry boxes for input features
drawing = st.number_input("Drawing", min_value=0.0, format="%.2f")
bright_annealing = st.number_input("Bright Annealing", min_value=0.0, format="%.2f")
sinking = st.number_input("Sinking", min_value=0.0, format="%.2f")
electro_fission = st.number_input("Electro Fission", min_value=0.0, format="%.2f")
# Button to trigger the prediction
if st.button("Predict"):
    # Prepare the input features for prediction
    input_features = np.array([[drawing, bright_annealing, sinking, electro_fission]])
    # Make the prediction using the loaded model
    prediction = model.predict(input_features)[0]
    # Display the result
if prediction == 1:  # Assuming 1 means Defected
       st.error("The Cannula is Distorted: Defected")
else: # Assuming 0 means Not Defected
       st.success("The Cannula is Not Distorted: No Defect")
