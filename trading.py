import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import joblib


feature_sets = {
		1:['Lag1_AAL','Lag2_AAL','Lag3_AAL','Lag4_AAL','Lag5_AAL',    'Lag1_WTI','Lag2_WTI','Lag3_WTI','Lag4_WTI','Lag5_WTI',     '5SMA', '20SMA','Volume'],
		2:['Lag1_AAL','Lag2_AAL','Lag3_AAL','Lag4_AAL','Lag5_AAL'],
		3:['Lag1_WTI','Lag2_WTI','Lag3_WTI','Lag4_WTI','Lag5_WTI'],
		4:['5SMA', '20SMA','Volume'],
		5:['5SMA', '20SMA','Lag1_AAL','Lag2_AAL','Lag3_AAL','Lag4_AAL','Lag5_AAL'],
		6:['5SMA', '20SMA','Lag1_WTI','Lag2_WTI','Lag3_WTI','Lag4_WTI','Lag5_WTI'],
	}





def main():

	result = {}

	#get true test dataset (maybe 2023 data)
	df = pd.read_csv('data/aal_features.csv')

	for k in range(1,7): #cycle through different feature sets
		for kernel in ['linear','rbf']:
			model_title = 'models/svm_' + str(k) + '_' + kernel + '_model.pkl' #fetch model title

			clf = joblib.load(model_title)
			scaler = joblib.load('scalers/svm_' + str(k) + '_' + kernel + '_scaler.pkl')

			#create predictions
			df = test_featureset(k, df, clf, scaler)

			returns = test(df['Close'], df['Predicted_Close'])

			result[model_title] = returns

	#print returns for each model
	for item in result:
		print(item, result[item])







def test_featureset(key, df, clf, scaler):
	'''test an individual set of features (tests a feature set on the entire dataframe's worth of days)'''
	def custom_predict(row):
		features = {feature: row[feature] for feature in feature_sets[key]}
		
		prediction = predict(features, clf, scaler)
		return prediction

	df['Predicted_Close'] = df.apply(custom_predict, axis=1)
	return df




def predict(features, clf, scaler):
	'''predict a single day's Close price based on a specific model/feature set'''
	features = pd.DataFrame([features])

	new_scaled_data = scaler.transform(features)

	#predict next day's price
	next_day_prediction = clf.predict(new_scaled_data)
	return next_day_prediction





def stock_return(real, pred): 
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


