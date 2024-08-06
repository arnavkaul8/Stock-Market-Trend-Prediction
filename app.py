import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import models
import pickle
import yfinance as yf
import streamlit as st
from sklearn import metrics
import requests
import datetime



import requests

def getTicker(company_name):
    yfinance = "https://query2.finance.yahoo.com/v1/finance/search"
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    params = {"q": company_name, "quotes_count": 1, "country": "United States", "enableFuzzyQuery": "true"}

    res = requests.get(url=yfinance, params=params, headers={'User-Agent': user_agent})
    data = res.json()  
    company_code = data['quotes'][0]['symbol']
    return company_code
 




def getStockName(symbol):
    comp = yf.Ticker(symbol)
    company_name = comp.info['longName']
    return company_name


def predict_test(model,x_test):
    y_predicted = model.predict(x_test)
    return y_predicted

def final_graph(model,y_predicted,model_name): 
    sub_header = "Original Price VS Predicted Price for " + model_name
    st.subheader(sub_header)
    fig = plt.figure(figsize=(12,6))
    plt.plot(y_test,'b', label = 'Original Price')
    plt.plot(y_predicted,'orange', label = 'Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig)



st.title('Stock Price Prediction')

user_input = st.text_input('Enter Stock Name', 'RELIANCE')
stock_ticker = getTicker(user_input)

stock_name = getStockName(stock_ticker)
original_name = 'stock name is : ' + f'<p style="font-family:Courier; color:Blue; font-size: 20px;">{stock_name}</p>'
st.markdown(original_name, unsafe_allow_html=True)

start = st.date_input(
    "Select Start Date for Analysis",
    datetime.date(2013, 1, 1))
end = st.date_input(
    "Select end Date for Analysis",
    datetime.date(2023, 1, 1))


df = yf.download(stock_ticker,start,end)
df = df.drop(columns = ['Adj Close'],axis=1)

# describing data
range_date = 'Data from '+str(start)+" to "+ str(end)
st.subheader(range_date)
st.write(df.describe())

#visualization
st.subheader('Closing Price VS Time Chart')
fig = plt.figure(figsize = (12,6))
plt.xlabel('Time')
plt.ylabel('Price')
plt.plot(df.Close,'green',label = 'Closing Price')
plt.legend()
st.pyplot(fig)
mavg_days=10

sub_head='Closing Price VS Time Chart with '+str(mavg_days)+' days moving average'
st.subheader(sub_head)

mavg = df.Close.rolling(mavg_days).mean()
fig = plt.figure(figsize = (12,6))
plt.xlabel('Time')
plt.ylabel('Price')
plt.plot(df.Close,'green',label = 'Closing Price')
plt.plot(mavg,'red',label = str(mavg_days)+' MVA')
plt.legend()
st.pyplot(fig)

# splitting data into training and testing samples
data_training = pd.DataFrame(df['Close'][:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])

# scaling down the data using Min-Max Scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))

data_training_array = scaler.fit_transform(data_training)

# Load the Model
LSTM_model = models.load_model('trained-model.h5')

# Load from file
pkl_filename = "stock-pred-SVR-model.pkl"
SVR_model = pickle.load(open(pkl_filename, 'rb'))


# predictions
past_mvavg_days = data_training.tail(mavg_days)
final_df = past_mvavg_days.append(data_testing, ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(mavg_days, input_data.shape[0]):
    x_test.append(input_data[i-mavg_days:i])
    y_test.append(input_data[i, 0])

x_test = np.array(x_test)
y_test = np.array(y_test)
x_test = x_test.squeeze()
y_test = scaler.inverse_transform(y_test.reshape(-1,1))

y_predicted_LSTM = predict_test(LSTM_model,x_test)
y_predicted_LSTM = scaler.inverse_transform(y_predicted_LSTM.reshape(-1,1))

x_test = x_test.squeeze()
y_predicted_SVM = predict_test(SVR_model,x_test)
y_predicted_SVM = SVR_model.predict(x_test)
y_predicted_SVM = scaler.inverse_transform(y_predicted_SVM.reshape(-1,1))


test_score_LSTM = metrics.mean_squared_error(y_test,y_predicted_LSTM)
st.write("Test Error of LSTM model = ",test_score_LSTM)
final_graph(LSTM_model,y_predicted_LSTM,"LSTM")
test_score_SVM = metrics.mean_squared_error(y_test,y_predicted_SVM)
st.write("Test Error of SVM model = ",test_score_SVM)
final_graph(SVR_model,y_predicted_SVM,"SVM") 




# forecasting
next_x_days = int(st.text_input('Enter number of days for prediction', '5'))
def predict_future_prices(model):
    closing_price_for_future=[]
    last_price = scaler.fit_transform(y_test)
    last_price = last_price.squeeze()
    total_len = last_price.shape[0]
    last_price = last_price[total_len-mavg_days:]

    for i in range(next_x_days):
        total_len = last_price.shape[0]
        test_data = last_price[total_len-mavg_days:]
        test_data = test_data.reshape((1,mavg_days))
        pred_price = model.predict(test_data)
        closing_price_for_future.append(pred_price)
        last_price = np.append(last_price,pred_price)

    closing_price_for_future = np.array(closing_price_for_future)
    closing_price_for_future = scaler.inverse_transform(closing_price_for_future.reshape(-1,1))
    return closing_price_for_future

future_SVM = predict_future_prices(SVR_model)
future_LSTM = predict_future_prices(LSTM_model)
closing_price_for_future = []
closing_price_for_future.append(future_SVM)
closing_price_for_future.append(future_LSTM)
closing_price_for_future = np.array(closing_price_for_future)
closing_price_for_future = closing_price_for_future.squeeze().T
predicted_table = pd.DataFrame(
    closing_price_for_future,
    columns=['SVM Price', 'LSTM Price'])
predicted_table.index = predicted_table.index+1
predicted_table.index.name = 'Days'
st.table(predicted_table)