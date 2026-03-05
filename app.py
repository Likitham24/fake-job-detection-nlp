import streamlit as st
import pickle
import pandas as pd

# Load trained model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Fake Job Detection App")
st.write("Enter a job description to check if it is Fake or Real.")

user_input = st.text_area("Job Description")

if st.button("Predict"):
    if user_input:
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)

        if prediction[0] == 1:
            st.error("⚠ This job posting is FAKE")
        else:
            st.success("✅ This job posting is REAL")
    else:
        st.warning("Please enter a job description.")
        
        
        