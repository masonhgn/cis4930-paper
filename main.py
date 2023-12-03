import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# Load the data
aal_df = pd.read_csv('moving_avgs.csv')

# Assuming 'Date' column exists in your CSV
aal_df['Date'] = pd.to_datetime(aal_df['Date'])

# Split the data into features, labels, and dates
X = aal_df[['20SMA', '5SMA']]
y = aal_df['Close']  # Changed to series
dates = aal_df['Date']

# Split the dataset into training and testing sets, including dates
X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(X, y, dates, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVR model
clf = SVR(kernel='linear')
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate MSE and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Cross-validation scores
scores = cross_val_score(clf, X, y, cv=5)
print("Cross-validated scores:", scores)
print("Average cross-validation score:", scores.mean())

# Plotting actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.plot(dates_test, y_test, label='Actual Prices', color='blue', marker='o')
plt.plot(dates_test, y_pred, label='Predicted Prices', color='red', marker='x')

plt.title('Actual vs Predicted Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




'''
def main():
	aal_df = pd.read_csv('AAL.csv')
	wti_df = pd.read_csv('WTI.csv')

	aal_df = aal_df.rename(columns={'Open': 'Open_AAL','Lag1':'Lag1_AAL','Lag2':'Lag2_AAL','Lag3':'Lag3_AAL','Lag4':'Lag4_AAL','Lag5':'Lag5_AAL'})
	wti_df = wti_df.rename(columns={'Open': 'Open_WTI','Lag1':'Lag1_WTI','Lag2':'Lag2_WTI','Lag3':'Lag3_WTI','Lag4':'Lag4_WTI','Lag5':'Lag5_WTI'})


	merged_df = aal_df.merge(wti_df, on='Date', how='inner')
	merged_df['Open_AAL'] = merged_df['Open_AAL'].shift(-1)
	merged_df = merged_df.dropna()
	#X = merged_df[['Lag5_AAL', 'Open_WTI', 'Lag1_WTI', 'Lag2_WTI', 'Lag3_WTI', 'Lag4_WTI', 'Lag5_WTI']]
	X = merged_df[['Lag1_AAL']]
	y = merged_df['Open_AAL']
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)


	clf = SVR(kernel='linear')

	clf.fit(X_train, y_train)

	y_pred = clf.predict(X_test)

	from sklearn.metrics import mean_squared_error, r2_score

	mse = mean_squared_error(y_test, y_pred)
	r2 = r2_score(y_test, y_pred)

	print(f"Mean Squared Error: {mse}")
	print(f"R-squared: {r2}")

	y_pred_all = clf.predict(X)

	dates = pd.to_datetime(merged_df['Date'])

	# Extract the actual and predicted opening prices
	actual_prices = merged_df['Open_AAL']
	predicted_prices = y_pred_all

	dates = pd.to_datetime(merged_df['Date'])

	# Define the number of x-axis ticks you want to display
	num_ticks = 5  # Adjust this number as needed

	# Create an array of indices to select ticks evenly spaced
	tick_indices = np.linspace(0, len(dates) - 1, num_ticks, dtype=int)

	# Get the corresponding dates for the selected tick indices
	selected_dates = dates[tick_indices]

	# Create an array of numbers for the X-axis (e.g., 0, 1, 2, ...)
	x_values = np.arange(len(dates))

	# Plot the actual and predicted prices over time with selected dates
	plt.figure(figsize=(10, 6))
	plt.plot(x_values, actual_prices, label='Actual Open Prices', color='b')
	plt.plot(x_values, predicted_prices, label='Predicted Open Prices', color='r')
	plt.xlabel('Date')  # Set the x-axis label to 'Date'
	plt.ylabel('Price')
	plt.title('Actual vs. Predicted Opening Prices Over Time (SVM Regression)')
	plt.xticks(tick_indices, selected_dates, rotation=45)  # Set x-axis ticks as selected_dates with rotation
	plt.legend(loc='upper left')
	plt.grid(True)
	plt.show()

if __name__=="__main__":
	main()
'''