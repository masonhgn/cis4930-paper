import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import random
from datetime import datetime
import matplotlib.dates as mdates

def direction_prediction(real, pred, price=True):
    pred_dirs = []

    for i in range(1, len(real)):
        if pred[i] > real[i-1]:
            pred_dirs.append(1)
        else:
            pred_dirs.append(-1)

    return pred_dirs

def test_performance(pred_dirs, real_price, initial=100000):
    capital = initial
    yesterday = real_price[0]
    status = 0
    earnings = []
    yields = []

    for i in range(1, len(real_price)):
        prev_capital = capital
        capital_ratio = 1
        if pred_dirs[i-1] == 1 and status != 1:
            status = 1
            capital_ratio = real_price[i] / yesterday
        elif pred_dirs[i-1] == -1 and status != -1:
            status = -1
            capital_ratio = yesterday / real_price[i]

        capital *= capital_ratio
        earnings_percent = (capital / initial * 100) - 100
        earnings.append(earnings_percent)
        yield_percent = (capital / prev_capital * 100) - 100
        yields.append(yield_percent)

        yesterday = real_price[i]


    return capital, yields, earnings


def random_performance(testing_vals, dates):
    random_pred_dirs = []
    for i in range(len(testing_vals)):
        random_pred_dirs.append(random.choice([-1, 1]))

    capital, yields, earnings = test_performance(random_pred_dirs, testing_vals)
    plot_with_dates(dates[1:], earnings, 'Earnings (%)', 'Random Guess Earnings')

    return

def plot_with_dates(dates, y, ylab, title):
    from datetime import datetime
    import matplotlib.dates as mdates

    dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
    plt.figure(figsize=(15, 5))
    plt.plot(dates, y)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=100))  # Adjust interval as needed
    plt.gcf().autofmt_xdate()  # Auto-rotate dates

    plt.xlabel('Date')
    plt.ylabel(ylab)
    plt.title(title)
    plt.show()

def data_preprocessing(raw_df, fts, target, split_by_date, split_date, training_percent):
    # Create a new dataframe with the given features
    new_df = pd.DataFrame(raw_df, columns=fts)

    # Get the dates
    dates = raw_df['Date'].values

    # Split the data by date
    if split_by_date == True:
        split_index = np.where(dates == split_date)[0][0]

        # Drop everything after split date
        new_df = new_df.drop(new_df.index[split_index:])
        dates = dates[:split_index]

    # Get the length of the data
    length = len(new_df)

    # Testing = last % of data 
    testing_df = new_df.loc[int(length*(1-training_percent)):]

    # Training = first % of data
    training_df = new_df.loc[:int(length*(1-training_percent))]

    # Transform testing into nparray
    testing_vals = testing_df[target].values

    # Transform training and testing into DMatrix
    dtrain = xgb.DMatrix(training_df.drop(target, axis=1), training_df[target])
    dtest = xgb.DMatrix(testing_df.drop(target, axis=1), testing_vals)

    return dtrain, dtest, testing_vals, dates[int(length*(1-training_percent)):], dates[:int(length*(1-training_percent))]

def xgboost_model(
    split_by_date = True,
    split_date = '2019-12-31',
    training_percent = 0.2,
    data_path = 'data/aal_features.csv',
    features = ['Close', 'Lag1_AAL', 'Lag2_AAL', 'Lag3_AAL', 'Lag4_AAL', 'Lag5_AAL'],
    target_feature = 'Close'
):
    # Data preprocessing
    stock_df = pd.read_csv(data_path)
    dtrain, dtest, testing_vals, test_dates, train_dates = data_preprocessing(stock_df, features, target_feature, split_by_date, split_date, training_percent)

    # Set the parameters for the xgboost
    param = {'max_depth':3, 'eta':1, 'objective':'reg:squarederror', 'eval_metric':'rmse'} 

    # Train the model
    num_rounds = 100
    model = xgb.train(param, dtrain, num_rounds)

    # Predict the model
    predictions = model.predict(dtest)

    # Calculate R2 and MSE
    rmse = model.eval(dtest, 'rmse')
    print('RMSE: %s' % rmse)

    # Generate Random guess performance
    random_performance(testing_vals, test_dates)

    # Plot the predictions and the actual values
    pred_dirs = direction_prediction(testing_vals, predictions)
    capital, yields, earnings = test_performance(pred_dirs, testing_vals)
    plot_with_dates(test_dates[1:], earnings, 'Earnings (%)', 'XGBoost Earnings')
    print('Capital: %s' % capital)
    print('Earnings: %s' % earnings[-1])

    # Plot the predictions and the actual values
    dates = [datetime.strptime(date, '%Y-%m-%d') for date in test_dates]
    plt.figure(figsize=(15, 5))
    plt.plot(dates, predictions, label='Predictions')
    plt.plot(dates, testing_vals, label='Actual Values')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=100))  # Adjust interval as needed
    plt.gcf().autofmt_xdate()  # Auto-rotate dates

    plt.xlabel('Date')
    plt.ylabel('AAL Price')
    plt.title('AAL Price Predictions vs Actual Values')
    plt.show()



if __name__ == '__main__':
    xgboost_model()