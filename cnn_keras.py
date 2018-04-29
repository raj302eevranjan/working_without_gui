from __future__ import print_function

import cv2 as cv
import numpy as np

from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
# from keras import backend as Kback

def build_model(imageShape, num_classes):
    model = Sequential()

    # Layer 1
    model.add(Conv2D(16, (3,3), activation='relu', padding="same", input_shape = imageShape))
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