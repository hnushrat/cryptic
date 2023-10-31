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

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


filename = "TrainingData.csv"
data = return_data(filename)


features = data.drop('class', axis=1)
features = features.values


labels = data['class'].values

xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size = 0.2, random_state = 42)


model = Sequential()

model.add(Dense(128, activation = 'relu', input_shape = xtrain.shape[1:]))
model.add(Dense(64, activation = 'relu'))

# model.add(Flatten())
model.add(Dense(8, activation ='relu'))
# model.add(Dropout(0.1))
# model.add(Flatten())
model.add(Dense(1, activation ='sigmoid'))

model.compile(loss = 'binary_crossentropy',
            optimizer='adam', metrics=['accuracy'])

print(model.summary())


best = keras.callbacks.ModelCheckpoint('best_classifier_dense.h5',monitor='val_accuracy')



model.fit(xtrain, ytrain, validation_data = (xtest, ytest), batch_size = 64, epochs = 100, callbacks = [best])
