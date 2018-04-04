import tensorflow as tf
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.activations import relu

model = Sequential()
model.add(Dense(1, input_dim=1, activation='sigmoid', use_bias=True))

X = np.array([1,1,0])
Y = np.array([1,1,0])

model.compile(optimizer = 'rmsprop', loss ="mean_squared_error", metrics = ['accuracy'])
model.fit(X, Y, epochs = 300, verbose = 2, batch_size = len(X), validation_split = .1)
print(model.predict(np.array([0])))
