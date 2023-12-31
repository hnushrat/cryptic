import pandas as pd
import numpy as np

import random

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, Dropout, Conv1D, Embedding

from transform_data import return_data
from get_encoded_vectors import return_encoded_vectors

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

filename = "TrainingData.csv"
data = return_data(filename)
features = return_encoded_vectors(filename)


features = np.expand_dims(features, axis = 1)


labels = data['class'].values

xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size = 0.2, random_state = 42)

model = Sequential()

model.add(LSTM(64, activation='tanh', return_sequences=True, input_shape = (1, 128)))
model.add(LSTM(64, activation='tanh', return_sequences=True))


model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))

model.compile(loss = 'binary_crossentropy',
            optimizer='adam', metrics=['accuracy'])

print(model.summary())


best = keras.callbacks.ModelCheckpoint('encoded_inputs_model.h5',monitor='val_accuracy')



model.fit(xtrain, ytrain, validation_data = (xtest, ytest), batch_size=64, epochs=100, callbacks=[best], verbose = 1)


model.save('encoded_inputs_last_epoch.h5')

print(f"\nSaving model at final epoch: as 'encoded_inputs_last_epoch.h5'\n")
