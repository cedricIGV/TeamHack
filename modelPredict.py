import keras
import numpy as np
from PIL import Image
from keras.models import Sequential, load_model
from keras.layers import Conv1D, GlobalMaxPooling1D, Activation, Dropout, Flatten, Dense, Reshape
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def main():
    model = load_model('inverted.h5')
    x = []
    x.append([1,0])
    x = np.array(x)
    x = np.reshape(x, (1,1,2))
    arr = model.predict(x)
    print(arr.shape)
    img = Image.fromarray(arr[0], 'L')
    img.save('out.bmp')

if __name__ == "__main__":
    main()
