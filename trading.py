import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import joblib
import math


def main():
	clf = joblib.load('svm_model.pkl')
	scaler = joblib.load('scaler.pkl')


	
	
	df = pd.read_csv('moving_avgs_new.csv')
	df['Predicted_Close'] = df.apply(lambda row: predict(row['20SMA'], row['5SMA'],clf,scaler), axis=1)

	print(test(df['Close'], df['Predicted_Close']))



def test(real, pred): 
    initial = 100000
    capital = initial
    yesterday = real[0]
    status = 0

    for i in range(1, len(real)):
        if pred[i] > yesterday and status != 1:
            status = 1
            capital = capital * (real[i] / yesterday)
        elif pred[i] < yesterday and status != -1:
            status = -1
            capital = capital * (yesterday / real[i])

        yesterday = real[i]

    return (capital - initial)/initial * 100






def predict(sma20, sma5, clf, scaler):
	new_data_for_today = pd.DataFrame({
	    '20SMA': [sma20],
	    '5SMA': [sma5]
	})

	#scale new data
	new_scaled_data = scaler.transform(new_data_for_today)

	#predict next day's price
	next_day_prediction = clf.predict(new_scaled_data)
	return next_day_prediction

if __name__=="__main__":
	main()





	'''
	# Initialize the first trade signal as NaN since there's no previous day to compare
	df['Trade_Signal'] = float('nan')

	# Generate trade signals (1 for Buy, 0 for Sell)
	for i in range(1, len(df)):
	    if df['Predicted_Close'].iloc[i] > df['Close'].iloc[i - 1]:
	        df['Trade_Signal'].iloc[i] = 1  # Buy
	    else:
	        df['Trade_Signal'].iloc[i] = 0  # Sell

	# The resulting DataFrame now has a column 'Trade_Signal' with buy/sell signals
	print(df[['Date', 'Close', 'Predicted_Close', 'Trade_Signal']])
	'''



	'''
		# Initialize trading variables
	cash_balance = 50000  # Starting cash balance
	stock_holding = 0     # Amount of stock held

	# Trading simulation
	for i in range(len(df)):
	    if df['Trade_Signal'].iloc[i] == 1 and cash_balance > 0:  # Buy
	        stock_holding = cash_balance / df['Close'].iloc[i]
	        cash_balance = 0
	    elif df['Trade_Signal'].iloc[i] == 0 and stock_holding > 0:  # Sell
	        cash_balance = stock_holding * df['Close'].iloc[i]
	        stock_holding = 0

	# Calculate final portfolio value
	final_portfolio_value = cash_balance + stock_holding * df['Close'].iloc[-1]

	# Print results
	print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
	'''


