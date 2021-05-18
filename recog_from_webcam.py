from PIL import Image as Pil
import cv2 as cv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model as lm
import numpy as np

def open(x):
    path = os.getcwd()
    img = Pil.open(os.path.join(path, x))
    img = img.resize((30, 30))
    img = np.expand_dims(img, axis=0)
    img = np.array(img)
    return img

img = open('sign.jpg')
model = lm('model.h5')
res = model.predict(img)
print(f"Розпізнаний знак: {np.argmax(res)}")