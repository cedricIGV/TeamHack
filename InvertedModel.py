import os
import glob
from skimage import io
import keras
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv1D, GlobalMaxPooling1D, Activation, Dropout, Flatten, Dense, Reshape
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import sys

def main():
    y = []
    x = []
    print("Loading Images...")
    total = len(os.listdir(os.path.join(os.getcwd(), "dog")))
    i = 1
    for filename in os.listdir(os.path.join(os.getcwd(), "dog")):
        path = os.path.join(os.getcwd(), "dog", filename)
        y.append(img_to_array(Image.open(path).convert('L')))
        x.append([1,0])
        if i % int(total/100) == 0:
            print("Loading Images " + str(int((i*50)/total)) + "%", end='\r'),
        i += 1

    for filename in os.listdir(os.path.join(os.getcwd(), "cat")):
        path = os.path.join(os.getcwd(), "cat", filename)
        y.append(img_to_array(Image.open(path).convert('L')))
        x.append([0,1])
        if i % int(total/100) == 0:
            print("Loading Images " + str(int((i*50)/total)) + "%", end='\r'),
        i += 1
        
    print("Image Loading Complete")
    print("Compiling Model...")
    model = Sequential()
    model.add(Dense(32, input_shape = (1, 2)))
    model.add(Activation('sigmoid'))
    model.add(Dense(640, activation='relu'))
    model.add(Dense(22500, activation='relu'))
    model.add(Reshape((150,150)))

    model.add(Conv1D(150, 3))
    model.add(Activation('relu'))
    model.add(GlobalMaxPooling1D())

    #model.add(Conv1D(150, 3))
    #model.add(Activation('relu'))
    #model.add(GlobalMaxPooling1D())

    model.add(Dense(22500, activation='relu'))
    model.add(Reshape((150,150)))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    print("Compiling Model Complete")
    print("Training Network...")
    i = 0
    while i < len(x)-100:
        y1 = np.reshape(np.array(y[i:i+100]), (100, 150, 150))
        x1 = np.reshape(np.array(x[i:i+100]), (100, 1, 2))

        model.fit(x1,y1, batch_size=10, epochs=10, verbose=2)
        model.save('inverted.h5')
        i = i+100
        print("Training Model " + str(int((i*100)/len(x))) + "%", end='\r'),

    print("Network Saved!")
if __name__ == "__main__":
    main()
