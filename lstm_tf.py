import pandas as pd
import numpy as np

import random

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, Dropout, Conv1D, Embedding


seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

data = pd.read_csv('TrainingData.csv')

for i in range(64):
    data[f"pos_{i+1}"] = np.nan


split = []
for i in data["Bitstream"].values:
    split.append(list(i))

data.iloc[:, 3:] = split

data.drop(["CID", "Bitstream"], axis = 1, inplace = True)


features = data.drop('class', axis=1)

features = features.values
features = np.expand_dims(features, axis = 1)

labels = data['class'].values

xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size = 0.2, random_state = 42)


model = Sequential()

model.add(LSTM(32, activation='tanh', return_sequences=True, input_shape = (1, 64)))
model.add(LSTM(32, activation='tanh', return_sequences=True))

model.add(Flatten())
model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss = 'binary_crossentropy',
            optimizer='adam', metrics=['accuracy'])

print(model.summary())

# Train the model on all available devices.

best = keras.callbacks.ModelCheckpoint('best_classifier_2.h5',monitor='val_accuracy')

model.fit(xtrain, ytrain, validation_data = (xtest, ytest), batch_size=64, epochs=100, callbacks=[best])