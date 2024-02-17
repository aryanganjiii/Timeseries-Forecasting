

import tensorflow as tf
import os
import numpy as np
import pandas as pd
from datetime import datetime

df = pd.read_csv('Vibration-Paya-line1-new1.csv')
df

df.index = pd.to_datetime(df['dateTime'], format='%Y-%m-%dT%H:%M:%SZ')

temp = df['values']
temp.plot()

def df_to_X_y(df, window_size=5):
  df_as_np = df.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [[a] for a in df_as_np[i:i+window_size]]
    X.append(row)
    label = df_as_np[i+window_size]
    y.append(label)
  return np.array(X), np.array(y)

WINDOW_SIZE = 1
X1, y1 = df_to_X_y(temp, WINDOW_SIZE)
X1.shape, y1.shape

X_train1, y_train1 = X1[:2300], y1[:2300]
X_val1, y_val1 = X1[2300:2500], y1[2300:2500]
X_test1, y_test1 = X1[2500:], y1[2500:]
X_train1.shape, y_train1.shape, X_val1.shape, y_val1.shape, X_test1.shape, y_test1.shape

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

bias_value = 0.1

model1 = Sequential()
model1.add(InputLayer((1, 1)))
model1.add(LSTM(80, return_sequences=True, bias_initializer=tf.keras.initializers.Constant(bias_value)))
model1.add(Dropout(0.3))
model1.add(LSTM(80, return_sequences=False, bias_initializer=tf.keras.initializers.Constant(bias_value)))
model1.add(Dense(40, activation='relu', bias_initializer=tf.keras.initializers.Constant(bias_value)))
model1.add(Dropout(0.3))
model1.add(Dense(1, activation='linear', bias_initializer=tf.keras.initializers.Constant(bias_value)))

model1.summary()


cp1 = ModelCheckpoint('model1/', save_best_only=True)
model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.00001), metrics=[RootMeanSquaredError()])

"""Epoch 25/25"""
"""13/13 [==============================] - 7s 545ms/step - loss: 0.3562 - root_mean_squared_error: 0.5969 - val_loss: 0.0249 - val_root_mean_squared_error: 0.1578"""

history=model1.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=80, callbacks=[cp1])



train_predictions = model1.predict(X_train1).flatten()
print(len(train_predictions))
train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train1})
train_results

import matplotlib.pyplot as plt
plt.plot(train_results['Train Predictions'])
plt.plot(train_results['Actuals'])
plt.show()






