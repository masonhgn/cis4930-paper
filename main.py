import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import joblib



def build_svm():
	# Load the original data
	aal_df = pd.read_csv('moving_avgs.csv')

	# Convert 'Date' to datetime
	aal_df['Date'] = pd.to_datetime(aal_df['Date'])

	# Split the data into features, labels, and dates
	X = aal_df[['20SMA', '5SMA']]
	y = aal_df['Close']
	dates = aal_df['Date']

	# Split the dataset into training and testing sets, including dates
	X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(X, y, dates, test_size=0.2, random_state=42)

	# Standardize the features
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	# Train the SVR model
	clf = SVR(kernel='linear')
	clf.fit(X_train_scaled, y_train)

	joblib.dump(clf, 'svm_model.pkl')

	# Save the scaler
	joblib.dump(scaler, 'scaler.pkl')




	# Predict on the test set
	y_pred = clf.predict(X_test_scaled)

	# Calculate MSE and R-squared for the original test data
	mse = mean_squared_error(y_test, y_pred)
	r2 = r2_score(y_test, y_pred)
	print(f"Original Data - Mean Squared Error: {mse}")
	print(f"Original Data - R-squared: {r2}")

	# Cross-validation scores on the original data
	scores = cross_val_score(clf, X, y, cv=5)
	print("Cross-validated scores:", scores)
	print("Average cross-validation score:", scores.mean())

	# Plotting actual vs predicted prices on the original data
	plot_df = pd.DataFrame({'Date': dates_test, 'Actual': y_test, 'Predicted': y_pred})
	plot_df.sort_values(by='Date', inplace=True)
	plt.figure(figsize=(12, 6))
	plt.plot(plot_df['Date'], plot_df['Actual'], label='Actual Prices', color='blue')
	plt.plot(plot_df['Date'], plot_df['Predicted'], label='Predicted Prices', color='red')
	plt.title('Actual vs Predicted Prices Over Time (Original Data)')
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.legend()
	plt.xticks(rotation=45)
	plt.tight_layout()
	plt.show()







def main():
	build_svm()

if __name__=="__main__":
	main()
