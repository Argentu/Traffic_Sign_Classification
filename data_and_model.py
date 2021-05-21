import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import cv2 as cv
import tensorflow.keras as tfk
from PIL import Image as Pil
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Activation

img_norm_width = 30
img_norm_height = 30
norm_size_tuple = img_norm_height, img_norm_width
num_cat = 43
pool_size = (2, 2)
model_input_shape = img_norm_width, img_norm_height, 3


def load_dataset(data_dir):
    data = []
    labels = []
    for i in range(num_cat):
        path = os.path.join(data_dir, "GTSRB", '000' + str(i))
        images = os.listdir(path)
        for j in images:
            try:
                image = cv.imread(os.path.join(path, j))
                image_from_array = Pil.fromarray(image, 'RGB')
                resized_image = image_from_array.resize(norm_size_tuple)
                data.append(np.array(resized_image))
                labels.append(i)
            except AttributeError:
                print("Can't upload images")
    img_train_data = data, labels
    return img_train_data


def create_model():
    model = Sequential()
    chDimension = -1

    model.add(Conv2D(8, (5, 5), padding="same", input_shape=(30, 30, 3)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chDimension))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chDimension))
    model.add(Conv2D(16, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chDimension))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chDimension))
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chDimension))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.7))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.7))

    model.add(Dense(43))
    model.add(Activation("softmax"))

    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

    return model
