import numpy as np
import pandas as pd
import streamlit as st
import sklearn
print("Scikit-learn version:", sklearn.__version__)
from sklearn.preprocessing import StandardScaler
import pickle

model = pickle.load(open("Customer_churn.pkl","rb"))
st.title("Customer Churn Prediction")
st.write('---')
st.header('Enter the Features of the Customer data:')
age=st.number_input('Age',min_value=18,max_value=70)
gender=st.selectbox('Gender:',['Male','Female'])
location=st.selectbox('Location:',['Houston','Los Angeles','Miami','Chicago','New York'])
subscription_length_months=st.number_input('Subscription Length (in months)',min_value=1,max_value=24)
monthly_bill=st.number_input('Monthly Bill',min_value=30.0,max_value=100.0)
total_usage_gb=st.number_input('Total Usage (in gb)',min_value=50.0,max_value=500.0)

def Predict(age,gender,location,subscription_length_months,monthly_bill,total_usage_gb):

    df = pd.DataFrame([[age,gender,location,subscription_length_months,monthly_bill,total_usage_gb]], columns=["Age", "gender", "location", "Subscription_Length_Months", "Monthly_Bill", "Total_Usage_GB"])
    
    # df["gender_female"] = df["gender"].apply(lambda x: 1 if x=="Female" else 0)
    df["Gender_Male"] = df["gender"].apply(lambda x: 1 if x=="Male" else 0)

    df["Location_Houston"] = df["location"].apply(lambda x: 1 if x=="Houston" else 0)
    df["Location_Los Angeles"] = df["location"].apply(lambda x: 1 if x=="Los Angeles" else 0)
    df["Location_Miami"] = df["location"].apply(lambda x: 1 if x=="Miami" else 0)
    # df["location_chicago"] = df["location"].apply(lambda x: 1 if x=="Chicago" else 0)
    df["Location_New York"] = df["location"].apply(lambda x: 1 if x=="New York" else 0)

    df = df.drop("gender", axis =1)
    df = df.drop("location", axis =1)


    df[["Age", "Subscription_Length_Months", "Monthly_Bill", "Total_Usage_GB"]] = StandardScaler().fit_transform(df[["Age", "Subscription_Length_Months", "Monthly_Bill", "Total_Usage_GB"]])

    prediction = model.predict(df)

    return prediction

if st.button('Predict Customer Churn'):
    Results = Predict(age,gender,location,subscription_length_months,monthly_bill,total_usage_gb)
    if Results[0] == 0:
        st.success("Hurray! No Customer Churn")
    elif Results[0] == 1:
        st.success("OOPS! Customer Churn")