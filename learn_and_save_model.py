from data_and_model import load_dataset, create_model
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.keras as tfk, numpy as np
from sklearn.model_selection import train_test_split

images, labels = load_dataset(os.path.dirname('data_and_model.py'))

labels = tfk.utils.to_categorical(labels)
x_train, x_test, y_train, y_test = train_test_split(
    np.array(images), np.array(labels), test_size=.2
)

model = create_model()
model.summary()

model.fit(x_train, y_train, batch_size=15, epochs=50, validation_split=.2)
model.evaluate(x_test, y_test)
model.save('test_model.h5')
