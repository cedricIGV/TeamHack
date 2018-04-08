import os
import glob
from skimage import io
import keras
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, Reshape
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


y = []
x = []
for filename in os.listdir(os.path.join(os.getcwd(), "dog")):
    path = os.path.join(os.getcwd(), "dog", filename)
    y.append(img_to_array(Image.open(path).convert('L')))
    x.append([1,0])

for filename in os.listdir(os.path.join(os.getcwd(), "cat")):
    path = os.path.join(os.getcwd(), "cat", filename)
    y.append(img_to_array(Image.open(path).convert('L')))
    x.append([0,1])

y = np.reshape(np.array(y), (172, 150, 150))
x = np.reshape(np.array(x), (172, 1, 2))

model = Sequential()
print(x.shape)
model.add(Dense(32, input_shape = (1, 2)))
model.add(Activation('sigmoid'))
model.add(Dense(640, activation='relu'))
model.add(Dense(22500, activation='relu'))
model.add(Reshape((150,150)))

#model.add(Conv2D(64, (3,3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="tf"))

#model.add(Conv2D(150, (3,3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="tf"))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(x,y, batch_size=50, epochs=5, verbose=2)
model.save_weights('inverted.h5')
