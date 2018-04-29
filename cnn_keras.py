from __future__ import print_function

import cv2 as cv
import numpy as np

from skimage.transform import resize
from datetime import datetime
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
# from keras import backend as Kback

def build_model(hight, weight, num_classes):
    model = Sequential()

    # Layer 1
    model.add(Conv2D(16, (3,3), activation='relu', padding="same", input_shape = (hight, weight, 1)))
    model.add(MaxPooling2D())
    
    model.add(Dropout(0.2))
    # Layer 2
    model.add(Conv2D(32, (5,5), padding="same"))
    model.add(MaxPooling2D())
    model.add(LeakyReLU(alpha=0.03))

    model.add(Dropout(0.25))
    # Layer 3
    model.add(Conv2D(32, (7,7), padding="same"))
    model.add(MaxPooling2D())
    model.add(LeakyReLU(alpha=0.03))

    model.add(Dropout(0.5))
    # Fully Connected Layer
    
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    return model


def shuffle(x, y):
    # TODO: Shuffle
    pass

def get_data(imageShape):
    benign_file = open('dataset/benign.txt')
    malignant_file = open('dataset/malignant.txt')
    normal_file = open('dataset/normal.txt')

    benign = [line.strip() for line in benign_file.readlines()]

    malignant = [line.strip() for line in malignant_file.readlines()]

    normal = [line.strip() for line in normal_file.readlines()]

    benign_img = [ cv.imread('{}.pgm'.format(no)) for no in benign ]
    benign_label = [ [0,1,0] for _ in xrange(len(benign)) ]

    malignant_img = [ cv.imread('{}.pgm'.format(no)) for no in malignant ]
    malignant_label = [ [0,0,1] for _ in xrange(len(malignant)) ]

    normal_img = [ cv.imread('{}.pgm'.format(no)) for no in normal ]
    normal_label = [ [1,0,0] for _ in xrange(len(normal)) ]

    x = []
    x.extend(benign_img)
    x.extend(malignant_img)
    x.extend(normal_img)

    # Reshaping image
    # x = [ cv.resize(img, imageShape) for img in x ]
    x =  [ resize(img, imageShape) for img in x ]

    y = []
    y.extend(benign_label)
    y.extend(malignant_label)
    y.extend(normal_label)

    # TODO: Shuffle

    x_train, x_test = x[:int(len(x)*0.7)], x[int(len(x)*0.3):] 
    y_train, y_test = y[:int(len(x)*0.7)], y[int(len(x)*0.3):] 

    x_train = np.array(x_train, dtype = np.float32) / 255
    y_train = np.array(y_train, dtype = np.float32)
    x_test = np.array(x_test, dtype = np.float32) / 255
    y_test = np.array(y_test, dtype = np.float32)

    return (x_train, y_train), (x_test, y_test)

# ---------------- Training -------------------------


start = datetime.now()

checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)

imageShape = (70, 70)
model = build_model(imageShape[0], imageShape[1], 3)
print('Done Building Model...')
(x_train, y_train), (x_test, y_test) = get_data(imageShape)

# Hyperparameters
epochs = 50
batch_size = 32

model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adadelta(), metrics=['accuracy'])

print('Done Compiling Model...')

print('Training...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks= [ checkpointer ],
          validation_data= (x_test, y_test))
print('Training Completed')

score = model.evaluate(x_test, y_test, verbose=1)

end = datetime.now()

print('Time Took: {}'.format(str(end - start)))
print('Test Loss: {}'.format(score[0]))
print('Test Accuracy: {}'.format(score[1]))