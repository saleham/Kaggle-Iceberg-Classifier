from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras import optimizers
import numpy as np

def CNN1():

    input_1 = Input(shape=(75, 75, 3), name="image")
    input_2 = Input(shape=[2], name="angle")

    X = Conv2D(64, kernel_size=(3,3), activation='relu')((input_1))
    X = MaxPooling2D((2, 2))(X)
    X = Dropout(0.3)(X)

    X = Conv2D(128, kernel_size=(3, 3), activation='relu')((X))
    X = MaxPooling2D((2, 2))(X)
    X = Dropout(0.3)(X)

    X = Conv2D(128, kernel_size=(3, 3), activation='relu')((X))
    X = MaxPooling2D((2, 2))(X)
    X = Dropout(0.3)(X)

    X = Conv2D(64, kernel_size=(3, 3), activation='relu')((X))
    X = MaxPooling2D((2, 2))(X)
    X = Dropout(0.2)(X)
    X = GlobalMaxPooling2D()(X)

    img_concat = (Concatenate()([X, BatchNormalization()(input_2)]))

    dense_ayer = Dropout(0.3)(BatchNormalization()(Dense(512, activation='relu')(img_concat)))
    dense_ayer = Dropout(0.2)((Dense(256, activation='relu')(dense_ayer)))
    output = Dense(1, activation="sigmoid")(dense_ayer)

    model = Model([input_1, input_2], output)

    #### compile model
    #sgd = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    return model

def VGG16():
    from keras.applications import VGG16
    input_1 = Input(shape=(75, 75, 3), name="image")
    input_2 = Input(shape=[2], name="angle")

    base_model = VGG16(weights='imagenet', include_top=False,
                       input_shape=(75, 75, 3), classes=1)
    base_model.trainable = False

    x = base_model(input_1)
    x = GlobalMaxPooling2D()(x)

    img_concat = (Concatenate()([x, BatchNormalization()(input_2)]))

    dense_ayer = Dropout(0.3)(BatchNormalization()(Dense(512, activation='relu')(img_concat)))
    dense_ayer = Dropout(0.3)((Dense(512, activation='relu')(dense_ayer)))
    output = Dense(1, activation="sigmoid")(dense_ayer)

    model = Model([input_1, input_2], output)

    sgd = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model