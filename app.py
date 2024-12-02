import streamlit as st
import numpy as np
import pickle
# Load the pre-trained model (make sure you have saved your model as 'saved_model.pkl')
# Uncomment the next line if you have a model file
model = pickle.load(open('saved_steps.pkl', 'rb'))
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
    # Uncomment the following line when the model is loaded
    # prediction = model.predict(input_features)[0]
    # Dummy prediction for demonstration purposes
    prediction = np.random.choice(["Defected", "Not Defected"])  # Replace with model prediction
    # Display the result
if prediction == "Defected":
st.error("The Cannula is Distorted: Defected")
else:
st.success("The Cannula is Not Distorted: No Defect")
