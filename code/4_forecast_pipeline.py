
import numpy as np
import pandas as pd
import os

import math
import lightgbm as lgb

def preprocess_df(df):
    df['date'] = pd.to_datetime(df['date'])

    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['weekday'] = df['date'].dt.day_name()

    start_date = pd.to_datetime('2013-01-01', format='%Y-%m-%d')
    df['time_idx'] = (df['date'] - start_date).dt.days + 1

    df_month = df.groupby(['store', 'item', 'year', 'month'])['sales'].sum().reset_index()
    df_month['date'] = pd.to_datetime(df_month['year'].astype(str) + '-' + df_month['month'].astype(str), format='%Y-%m').dt.to_period('M')

    start_date = pd.to_datetime('2013-01', format='%Y-%m')
    df_month['time_idx'] = (df_month['date'].dt.year * 12 + df_month['date'].dt.month) - (start_date.year * 12 + start_date.month) + 1
    return df_month

def pretrain_df(df_month):
    df_pretrain = df_month.copy()

    df_pretrain = pd.get_dummies(df_pretrain, columns=['month'], prefix='month')
    df_pretrain = pd.get_dummies(df_pretrain, columns=['year'], prefix='year')

    df_pretrain[df_pretrain.select_dtypes(include=bool).columns] = df_pretrain.select_dtypes(include=bool).astype(int)
    df_pretrain.drop(columns=['date'], inplace=True)

    # Create lag features for time-series modeling
    num_lags = 12 
    for i in range(1, num_lags + 1):
        df_pretrain[f'nb_order_lag_{i}'] = df_pretrain['sales'].shift(i)

    df_pretrain = df_pretrain.dropna()
    return df_pretrain

def train_test_split(df_pretrain):
    # Separating target variable 'nb_order' and features
    y = df_pretrain['sales']
    X = df_pretrain.drop(columns='sales')

    # Scaling features and combining with the target variable
    df_scaled = pd.DataFrame(np.column_stack([y, X]), columns=['sales'] + X.columns.tolist())

    # Split the data into train and test sets
    train_size = int(len(df_scaled))  # Use all of data for train
    train, test = df_scaled[:train_size], df_scaled[train_size-1:]

    return train, test

def train_model(X_train, y_train):
    # Set hyperparameters
    model = lgb.LGBMRegressor(
        force_col_wise=True,
        learning_rate=0.1,
        n_estimators=100,
        min_data_in_leaf = 0
    )

    model.fit(X_train, y_train)
    return model

def forecast(model, X_test):
    # Initialize an empty list to store forecasted values
    forecast_values = []

    # Initialize the current sequence with the first row of the test set
    current_sequence = X_test[0:1]
    # Loop to predict the next 12 months
    for _ in range(12):
        next_month_pred = model.predict(current_sequence)
        time_idx = current_sequence[0][0] + 1
        month_seq = current_sequence[0][1:13] 
        year_seq = current_sequence[0][13:18]

        # Check if the last element of month_seq is 1 and swap values if true
        if month_seq[-1] == 1:
            month_seq[0], month_seq[-1] = 1, 0
        else:
            month_seq = np.roll(month_seq, 1)

        current_sequence_lag = current_sequence[0][18:]
        current_sequence_lag = np.roll(current_sequence_lag, 1)
        current_sequence_lag[0] = next_month_pred[0]

        current_sequence = [np.concatenate(([time_idx], month_seq, year_seq, current_sequence_lag))]

        # Predict the next month's value using the trained model
        next_hour_pred = model.predict(current_sequence)
        
        # Append the prediction to the forecast values
        forecast_values.append(next_hour_pred[0])
        
    return forecast_values

def forecast_pipeline(body):
    df_month = preprocess_df(body)
    df_pretrain = pretrain_df(df_month)

    train, test = train_test_split(df_pretrain)
    X_train, y_train = train.drop(columns='sales').values, train['sales'].values
    X_test, y_test = test.drop(columns='sales').values, test['sales'].values

    model = train_model(X_train, y_train)
    prediction = forecast(model, X_test)
    # print(prediction)
    prediction = [math.ceil(num * 1.1) for num in prediction]


    return prediction

df = pd.read_csv(os.path.abspath(os.path.join("static", "../datasets/train.csv")))
body = df[(df['store']==1) & (df['item']==1)]
prediction = forecast_pipeline(body)
print(prediction)

# body = {
#     'date': '2013-01-01'
#     'sales': 13
# }

def service_forecast(body):
    return forecast_pipeline(body)