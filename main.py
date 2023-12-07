import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import joblib


def run_all_models():

	#these are all the different feature sets we will test to figure out which combinations of features are most effective
	feature_sets = {
		1:['Lag1_AAL','Lag2_AAL','Lag3_AAL','Lag4_AAL','Lag5_AAL',    'Lag1_WTI','Lag2_WTI','Lag3_WTI','Lag4_WTI','Lag5_WTI',     '5SMA', '20SMA','Volume'],
		2:['Lag1_AAL','Lag2_AAL','Lag3_AAL','Lag4_AAL','Lag5_AAL'],
		3:['Lag1_WTI','Lag2_WTI','Lag3_WTI','Lag4_WTI','Lag5_WTI'],
		4:['5SMA', '20SMA','Volume'],
		5:['5SMA', '20SMA','Lag1_AAL','Lag2_AAL','Lag3_AAL','Lag4_AAL','Lag5_AAL'],
		6:['5SMA', '20SMA','Lag1_WTI','Lag2_WTI','Lag3_WTI','Lag4_WTI','Lag5_WTI'],
	}


	'''run all models with the above features, and record which features used, mse and r2'''



	#svm
	kernels = ['linear','rbf']
	result = []
	for fs in feature_sets.keys():
		for k in kernels:
			print('running feature set ' + str(fs) + ' with kernel '+ k)
			model = build_svm(feature_sets[fs], fs, k)
			result.append(model)
			
	return result



def build_svm(feature_list, features_idx, kern):

	title = 'svm_'+ str(features_idx) + '_' + kern

	# Load the original data
	features = pd.read_csv('data/aal_features.csv')

	# Convert 'Date' to datetime
	features['Date'] = pd.to_datetime(features['Date'])

	# Split the data into features, labels, and dates
	X = features[feature_list]
	y = features['Close']
	dates = features['Date']
	# Split the dataset into training and testing sets, including dates
	X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(X, y, dates, test_size=0.2, random_state=42)

	# Standardize the features
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	# Train the SVR model
	clf = SVR(kernel=kern)
	clf.fit(X_train_scaled, y_train)

	joblib.dump(clf, 'models/' + title + '_model.pkl')

	# Save the scaler
	joblib.dump(scaler, 'scalers/' + title + '_scaler.pkl')




	# Predict on the test set
	y_pred = clf.predict(X_test_scaled)

	# Calculate MSE and R-squared for the original test data
	mse = mean_squared_error(y_test, y_pred)
	r2 = r2_score(y_test, y_pred)
	print(f"Original Data - Mean Squared Error: {mse}")
	print(f"Original Data - R-squared: {r2}")

	'''
	# Cross-validation scores on the original data
	scores = cross_val_score(clf, X, y, cv=3, n_jobs=-1)
	print("Cross-validated scores:", scores)
	print("Average cross-validation score:", scores.mean())
	'''
	# Plotting actual vs predicted prices on the original data
	plot_df = pd.DataFrame({'Date': dates_test, 'Actual': y_test, 'Predicted': y_pred})
	plot_df.sort_values(by='Date', inplace=True)
	plt.figure(figsize=(12, 6))

	plt.plot(plot_df['Date'], plot_df['Actual'], label='Actual Prices', color='blue')
	plt.plot(plot_df['Date'], plot_df['Predicted'], label='Predicted Prices', color='red')
	plt.title('Actual vs Predicted Prices Over Time')
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.legend()
	plt.xticks(rotation=45)

	#plt.tight_layout()

	plt.title('Features: (' + ', '.join(feature_list) + ') Kernel: ' + kern, wrap=True)

	plt.savefig('images/'+title)

	plt.close()

	return [features_idx,kern,mse,r2]
	#return [features_idx,kern,mse,r2,scores.mean()]





def main():
	vec = run_all_models()

	for row in vec: print(row)

if __name__=="__main__":
	main()
