import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os, time
from datetime import datetime
import yahoo_fin.stock_info as yf
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, BatchNormalization
import tensorflow as tf
TICKER = 'ETH-USD'
# data = yf.get_data(TICKER,start_date = '14/06/2014', end_date='15/06/2021')
# data.to_csv(TICKER + '.csv')
SEQ_LEN = 30
df = pd.read_csv(TICKER + '.csv')
df.rename(columns = {'Unnamed: 0': 'Date'}, inplace = True)
ax = plt.gca()
plt.xticks(rotation=45)

plt.plot(df['Date'], df["adjclose"], label = 'Adjusted Close')
x_min, x_max = ax.get_xlim()
ax.set_xticks(np.linspace(x_min, x_max, 10))
# plt.show()
# print(df)
df['Date'] = pd.to_numeric(pd.to_datetime(df['Date']))
filtered_df = pd.DataFrame(index = range(0, len(df)), columns = ['Date', 'Adj_Close'])
# print(filtered_df.columns)

for i in range(0, len(df.index)):
    filtered_df['Date'][i] = df['Date'][i]
    filtered_df['Adj_Close'][i] = df['adjclose'][i]
# print(df)
print(len(filtered_df))
train_range = round(len(filtered_df.index) * 0.8)

def Scale_Data():
    scaler = MinMaxScaler(feature_range = (0,1))
    dataset = filtered_df.values
    # print(dataset)
    #train on 80% of data, test on 20% 
    #train_range = round(len(filtered_df.index) * 0.8)
    train_data = dataset[0:train_range, :] 
    test_data = dataset[train_range:, :]
    # print(dataset[train_range:, :])
    filtered_df.drop("Date", axis=1, inplace=True)
    # print(filtered_df.head)
    # print(dataset)
    scaled_data = scaler.fit_transform(dataset)
    # print(scaled_data[:10])
    x_train, y_train = [], []
    for i in range(SEQ_LEN, len(train_data)):
        x_train.append(scaled_data[i-SEQ_LEN:i, 0])
        y_train.append(scaled_data[i, 0])
    # print(x_train[:10], '\n --------------------------')
    # print(y_train[:10])
    x_train, y_train = np.array(x_train),np.array(y_train)
    x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))
    
    return x_train, y_train, test_data, scaled_data

def Build_Model():
    model = Sequential()
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train, y_train, test_data, scaled_data = Scale_Data()
    x_test, y_test = [], []
    for i in range(train_range + SEQ_LEN, train_range+ len(test_data)):
        x_test.append(scaled_data[i-SEQ_LEN:i, 0])
        y_test.append(scaled_data[i, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1], 1))

    model.add(LSTM(128, return_sequences=True, input_shape = (x_train.shape[1], 1)))
    model.add(Dropout(0.5))

    # model.add(LSTM(128, return_sequences=True, input_shape = (x_train.shape[1], 1)))
    # model.add(Dropout(0.2))

    model.add(LSTM(128, input_shape = (x_train.shape[1], 1), return_sequences=True))
    model.add(Dropout(0.5))

    model.add(LSTM(128, input_shape = (x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    input_data = filtered_df[len(filtered_df) - len(test_data)-SEQ_LEN:].values
    input_data = input_data.reshape(-1,1)
    input_data = scaler.fit_transform(input_data)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=30,batch_size = 40, validation_data= (x_test, y_test))
    # print(scaled_data[train_range: train_range+10])
    # print('\n --------------------------', x_test[:10], '\n --------------------------')
    # print(y_test[:10])
    # print(test_data)

    X_test = []
    for i in range(SEQ_LEN, input_data.shape[0]):
        X_test.append(input_data[i-SEQ_LEN : i, 0])
    X_test = np.array(X_test)
    # print(input_data[:10])

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
    # print('x',X_test[:10])
    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)

    model.save('lstm.h5')

    return predicted_price

def Analysis():
    predicted_price = Build_Model()
    # print(filtered_df.columns)
    # filtered_df["Date"]=pd.to_datetime(filtered_df["Date"],format="%Y-%m-%d")
    train_data = filtered_df[:train_range]
    test_data = filtered_df[train_range:]

    test_data['Predicted'] = predicted_price
    print('test', test_data, "\n ---------------------------------")
    # print('train', train_data, "\n ---------------------------------")
    plt.plot(train_data['Adj_Close'], "-r", label = 'train')
    plt.plot(test_data['Adj_Close'], '-g', label = 'test')
    plt.plot(test_data['Predicted'], '-b', label = 'Predicted')
    plt.show()
Analysis()
    # x_train, y_train = np.array([]), np.array([])
# plt.figure(figsize=(16,8))
# plt.plot(df["adjclose"], label = 'Adjusted Close')
# plt.show()