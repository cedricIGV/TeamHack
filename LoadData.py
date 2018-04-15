import os
import glob
import keras
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def main():
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=.2,
        height_shift_range=.2,
        rescale=1./255,
        shear_range=.2,
        zoom_range=.2,
        horizontal_flip=True,
        fill_mode='nearest')

    #for filename in glob.glob(os.getcwd() + "/downloads/dog/*.jpg"):
    #    img = load_img(filename)
    #    x = img_to_array(img)
    #    x = x.reshape((1,) + x.shape)

    train_generator = datagen.flow_from_directory(os.getcwd() + "\downloads", batch_size=1, target_size=(150,150), save_to_dir='processed_img', save_format='jpg')

    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape=(150, 150, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="tf"))

    model.add(Conv2D(32, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="tf"))

    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="tf"))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(.5))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit_generator(train_generator, steps_per_epoch = 20, epochs=5, verbose=2)
    model.save_weights('classifier.h5')

    path = os.path.join(os.getcwd(), "dog")
    filenames = os.listdir(path)
    i = 0
    x = []
    for filename in filenames:
        if i == 50:
            break
        im = Image.open(os.path.join(path, filename))
        arr = np.array(im)
        x.append(arr)
        i += 1

    out = model.predict(np.array(x))
    print(out)

if __name__ == "__main__":
    main()
