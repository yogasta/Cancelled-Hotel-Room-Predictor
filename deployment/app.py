import streamlit as st
import pandas as pd
import joblib
import requests,io

st.header('Cancelled Hotel Reservation Prediction')
st.write("""
Created by Prayoga Agusto Haradi

Please use the sidebar to select input features.
""")

@st.cache
def fetch_data():
    url = 'https://raw.githubusercontent.com/H8-Assignments-Bay/p1-ftds004-hck-m2-yogasta/main/Hotel%20Reservations.csv'
    token = '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'

    headers = {'Authorization': f'token {token}'}
    response = requests.get(url, headers=headers)

    df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
    return df[['lead_time','avg_price_per_room','no_of_special_requests','market_segment_type','booking_status']]

df = fetch_data()
st.write(df)

st.sidebar.header('User Input Features')
st.subheader('Explanation of Features')
st.write("""
lead_time is the difference of day from the time of reservation to the arrival date of the guest.
""")
st.write("""
avg_price_per_room is the average price of the room each night.
""")
st.write("""
no_of_special_requests is the amount of special requests that each guest requested.
""")
st.write("""
market_segment_type is the segment type of said guest (Online, Offline, Corporate, etc.)
""")
st.write("""
""")


def user_input():
    lead_time = st.sidebar.number_input('lead_time', min_value=0,value=50)
    avg_price_per_room = st.sidebar.number_input('avg_price_per_room', min_value=0,value=80)
    no_of_special_requests = st.sidebar.number_input('no_of_special_requests', min_value=0,value=3)
    market_segment_type = st.sidebar.radio('What is the market segment of your guest?', ("Offline","Online","Corporate","Aviation","Complementary"))


    data = {
        'lead_time': lead_time,
        'avg_price_per_room': avg_price_per_room,
        'no_of_special_requests': no_of_special_requests,
        'market_segment_type': market_segment_type
    }
    features = pd.DataFrame(data, index=[0])
    return features


input = user_input()

st.subheader('Your Input')
st.write("""
Input your data for the model to predict.
""")
st.write(input)

features_pred = input[['lead_time','avg_price_per_room','no_of_special_requests','market_segment_type']]
load_model = joblib.load('KNN.pkl')
prediction = load_model.predict(features_pred)

if prediction == 1:
    prediction = 'The Guest is most likely to cancel their reservation (Canceled)'
else:
    prediction = 'The Guest is most likely to keep their reservation (Not Canceled)'

st.write('Based on your input, the placement model predicted: ')
st.write(prediction)