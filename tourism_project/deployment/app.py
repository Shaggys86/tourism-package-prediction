import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="Shaggys86/tourism_pkg_prediction_model", filename="best_tourism_pkg_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Visit with us - Tourism Package Prediction")
st.write("""
This application predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them.
This robust predictive solution will empower policymakers to make data-driven decisions, enhance marketing strategies, and effectively target potential customers, thereby driving customer acquisition and business growth.

Please provide the following information to make a prediction:
""")

# User input
#-----------------
age = st.number_input("Age", min_value=18, max_value=100, value=30)
type_of_contact = st.selectbox("Type of Contact", ["Company Invited","Self Enquiry"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1)
occupation = st.selectbox("Occupation", ["Free Lancer","Salaried", "Small Business", "Large Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
no_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10)
no_of_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10)
product_pitched = st.selectbox("Product Pitched",["Deluxe","Basic","Standard","King","Super Deluxe"])
prefrerred_property_star = st.number_input("Preferred Property Star", min_value=1, max_value=5)
marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced","Unmarried"])
no_of_trips = st.number_input("Number of Trips", min_value=1, max_value=100)
passport = st.selectbox("Passport (0: No, 1: Yes)", ["1", "0"])
pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5)
own_car = st.selectbox("Own Car (0: No, 1: Yes)", ["1", "0"])
no_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10)
designation = st.selectbox("Designation",["AVP","Executive","Manager","Senior Manager","VP"])
monthly_income = st.number_input("Monthly Income", min_value=0, max_value=1000000)
#------------------

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': type_of_contact,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': no_of_person_visiting,
    'NumberOfFollowups': no_of_followups,
    'ProductPitched': product_pitched,
    'PreferredPropertyStar': prefrerred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': no_of_trips,
    'Passport': passport,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': no_of_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income
}])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    st.success(f"Product taken (0: No, 1: Yes): **{prediction}**")
