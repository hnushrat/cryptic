import pandas as pd
import numpy as np

import random

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, Dropout, Conv1D

from transform_data import return_data

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

filename = "TrainingData.csv"
data = return_data(filename)

features = data.drop('class', axis=1)
features = features.values

print(features.shape)

model = Sequential()


model.add(Dense(128, activation='tanh', input_shape = features.shape[1:]))

model.add(Dense(features.shape[1], activation='sigmoid'))

model.compile(loss = 'binary_crossentropy',
              optimizer='adam')

print(model.summary())

best = keras.callbacks.ModelCheckpoint('best_encoder.h5',monitor='loss')



model.fit(features, features, batch_size=64, epochs=100, callbacks=[best])
