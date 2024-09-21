from flask import Flask, jsonify, request
import yfinance as yf
import datetime as dt
from prophet import Prophet

app = Flask(__name__)

# In-memory cache
cache = {
    'data': {},
    'last_fetch': {}  # Store last fetch time for each symbol
}

@app.route('/predict', methods=['GET'])
def predict():
    # Get the cryptocurrency symbol from the query parameter
    symbol = request.args.get('symbol', 'BTC-USD')  # Default to BTC-USD if no symbol is provided
    
    # Check if the cache is valid for the specific symbol
    today = dt.datetime.now().date()

    # Initialize the symbol in the cache if it doesn't exist
    if symbol not in cache['data']:
        cache['data'][symbol] = None
        cache['last_fetch'][symbol] = None

    if cache['last_fetch'][symbol] != today:
        # If the cache is invalid or not fetched today, fetch new data
        start = today - dt.timedelta(days=2*365)
        end = today
        data = yf.download(symbol, start=start, end=end)

        # Prepare the data for Prophet
        df = data.reset_index()[['Date', 'Close']]
        df.columns = ['ds', 'y']

        # Train the Prophet model
        model = Prophet(daily_seasonality=True)
        model.fit(df)

        # Make future predictions for 100 days
        future = model.make_future_dataframe(periods=100)
        forecast = model.predict(future)
        forecast = forecast.tail(200)
        df = df.tail(100)

        # Prepare the data for JSON response
        response_data = {
            'symbol': symbol,
            'dates': forecast['ds'].dt.strftime('%Y-%m-%d').tolist(),
            'predicted': forecast['yhat'].tolist(),
            'lower_bound': forecast['yhat_lower'].tolist(),
            'upper_bound': forecast['yhat_upper'].tolist(),
            'historical': df['y'].tolist()
        }

        # Update the cache for the specific symbol
        cache['last_fetch'][symbol] = today
        cache['data'][symbol] = response_data  # Cache the response for the specific symbol

    else:
        # Return cached data
        response_data = cache['data'][symbol]

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
