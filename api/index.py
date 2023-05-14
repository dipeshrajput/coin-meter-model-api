from flask import Flask, jsonify, request
from tensorflow import keras
import joblib
import numpy as np
from datetime import date, timedelta
import pandas as pd

sDate = date(2023,5,13)

# model_btc_open = keras.models.load_model('../models/btc_open_model')
model_btc_close = keras.models.load_model('../models/btc_close_model')
model_btc_high = keras.models.load_model('../models/btc_high_model')
# model_btc_low = keras.models.load_model('../models/btc_low_model')

# model_eth_open = keras.models.load_model('../models/eth_open_model')
model_eth_close = keras.models.load_model('../models/eth_close_model')
model_eth_high = keras.models.load_model('../models/eth_high_model')
# model_eth_low = keras.models.load_model('../models/eth_low_model')

# scaler_btc_open = joblib.load('../models/scaler/btc_open_scaler.gz')
scaler_btc_close = joblib.load('../models/scaler/btc_close_scaler.gz')
scaler_btc_high = joblib.load('../models/scaler/btc_high_scaler.gz')
# scaler_btc_low = joblib.load('../models/scaler/btc_low_scaler.gz')

# scaler_eth_open = joblib.load('../models/scaler/eth_open_scaler.gz')
scaler_eth_close = joblib.load('../models/scaler/eth_close_scaler.gz')
scaler_eth_high = joblib.load('../models/scaler/eth_high_scaler.gz')
# scaler_eth_low = joblib.load('../models/scaler/eth_low_scaler.gz')

test_btc_open = ''
test_btc_close = ''
test_btc_high = ''
test_btc_low = ''

test_eth_open = ''
test_eth_close = ''
test_eth_high = ''
test_eth_low = ''

with open('../models/npy/test_data.npy', 'rb') as f:
    test_btc_open = np.load(f)
    test_btc_close = np.load(f)
    test_btc_high = np.load(f)
    test_btc_low = np.load(f)
    test_eth_open = np.load(f)
    test_eth_close = np.load(f)
    test_eth_high = np.load(f)
    test_eth_low = np.load(f)

app = Flask(__name__)

def getData(days, model, scaler, np_data):
    x_input=np_data[600:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    lst_output=[]
    n_steps=100
    i=0
    while(i<days):
        if(len(temp_input)>100):
            #print(temp_input)
            x_input=np.array(temp_input[1:])
            # print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            # print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            # print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            # print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1
    return scaler.inverse_transform(lst_output)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/getBTCPred', methods=["GET"])
def getBTCPred():
    number_of_days = request.json['number_of_days']
    close = getData(number_of_days, model_btc_close, scaler_btc_close, test_btc_close)
    high = getData(number_of_days, model_btc_close, scaler_btc_close, test_btc_high)
    dates = pd.date_range(sDate, sDate+timedelta(days=number_of_days-1),freq='d').strftime("%Y-%m-%d").tolist()

    return jsonify({
        'close': close.tolist(),
        'high': high.tolist(),
        'dates': dates
    })

@app.route('/getETHPred', methods=["GET"])
def getETHPred():
    number_of_days = request.json['number_of_days']
    close = getData(number_of_days, model_eth_close, scaler_eth_close, test_eth_close)
    high = getData(number_of_days, model_eth_close, scaler_eth_close, test_eth_high)
    dates = pd.date_range(sDate, sDate+timedelta(days=number_of_days-1),freq='d').strftime("%Y-%m-%d").tolist()

    return jsonify({
        'close': close.tolist(),
        'high': high.tolist(),
        'dates': dates
    })

@app.route('/getPred', methods=["GET"])
def getPred():
    number_of_days = request.json['number_of_days']
    close_btc = getData(number_of_days, model_btc_close, scaler_btc_close, test_btc_close)
    high_btc = getData(number_of_days, model_btc_close, scaler_btc_close, test_btc_high)
    close_eth = getData(number_of_days, model_eth_close, scaler_eth_close, test_eth_close)
    high_eth = getData(number_of_days, model_eth_close, scaler_eth_close, test_eth_high)
    dates = pd.date_range(sDate, sDate+timedelta(days=number_of_days-1),freq='d').strftime("%Y-%m-%d").tolist()

    return jsonify({
        'close_btc': close_btc.tolist(),
        'high_btc': high_btc.tolist(),
        'close_eth': close_eth.tolist(),
        'high_eth': high_eth.tolist(),
        'dates': dates
    })

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')


