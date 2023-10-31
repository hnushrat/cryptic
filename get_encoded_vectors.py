import pandas as pd
import numpy as np

import random


import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, Dropout, Conv1D, Embedding

from transform_data import return_data


def return_encoded_vectors(filename):
    
    # seed = 42
    # random.seed(seed)
    # np.random.seed(seed)
    # tf.random.set_seed(seed)

    data = return_data(filename)

    features = data.drop('class', axis=1)

    features = features.values

    encoder_model = keras.models.load_model("best_encoder.h5")

    get_encoded_vector = Sequential()

    get_encoded_vector.add(Dense(128, activation='tanh', input_shape = features.shape[1:]))

    get_encoded_vector.set_weights(encoder_model.weights[:2])

    encoded_data = get_encoded_vector(features).numpy()

    return encoded_data
