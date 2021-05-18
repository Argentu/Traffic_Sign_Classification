import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2 as cv
import numpy as np
import tensorflow.keras as tfk
from PIL import Image as Pil
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dropout, Dense

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
        path = os.path.join(data_dir, "GTSRB", '000'+str(i))
        imgs_list = os.listdir(path)
        for j in imgs_list:
            try:
                img = Pil.open(os.path.join(path, j))
                normal_size_img = img.resize(norm_size_tuple)
                data.append(np.array(normal_size_img))
                labels.append(i)
            except AttributeError:
                print("Can't upload images")
    img_train_data = data, labels
    return img_train_data


def create_model():
    model = tfk.Sequential([

        Conv2D(8, (5, 5), padding='same', activation='relu', input_shape=model_input_shape),
        BatchNormalization(axis=-1),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(16, (3, 3), padding='same', activation='relu'),
        BatchNormalization(axis=-1),
        Conv2D(16, (3, 3), padding='same', activation='relu'),
        BatchNormalization(axis=-1),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(axis=-1),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(axis=-1),
        MaxPooling2D(pool_size=(2, 2)),

        #Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.7),

        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.7),

        Dense(num_cat, activation='softmax')
    ])

    model.compile(loss="categorical_crossentropy",
                  optimizer="nadam",
                  metrics=["accuracy"])

    return model
