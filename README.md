

### Comparative Analysis of Machine Learning Models for Predicting Airline Stock Price Movements

We have organized our three models which will all run in main.

#### Build Instructions
Assumptions: You have the latest version of Python and pip installed
1. Run ```pip install -r requirements.txt``` to get all of the dependencies.
2. Run ```python main.py```

The entire process of running all three models should take around 2 minutes or less.








Some details about xgboost:

Parameters for xgboost_model function:
-split_by_date = False  # If True, split the dataset by date into training and testing sets
-split_date = '2019-12-31' # Date to split the dataset on
-training_percent = 0.2 # Percentage of data to use for training
-data_path = 'cis4930-paper/data/aal_features.csv' # Path to the dataset
-features = ['Close', 'Lag1_AAL', 'Lag2_AAL', 'Lag3_AAL', 'Lag4_AAL', 'Lag5_AAL'] # Features to use for training, must include target feature
-target_feature = 'Close' # Target feature to predict

