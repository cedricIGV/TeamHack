import tensorflow as tf #Needed import statements
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.activations import relu

model = Sequential() #Determine model type
model.add(Dense(1, input_dim=1, activation='sigmoid', use_bias=True)) #add layers

X = np.array([1,1,0]) #Training Data
Y = np.array([1,1,0])

model.compile(optimizer = 'rmsprop', loss ="mean_squared_error", metrics = ['accuracy']) #Create model and set optimizer type
model.fit(X, Y, epochs = 300, verbose = 2, batch_size = len(X), validation_split = .1) #Train model
print(model.predict(np.array([0]))) #Print model ouput when given a 0 input
