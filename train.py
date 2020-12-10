import os
from random import shuffle

import tensorflow.compat.v1 as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

import numpy as np
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from config import Config
from data import get_training_name
from model import model_inceptionv3

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

def train_top(model_name):
    backbone, model = model_inceptionv3()

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in backbone.layers:
        layer.trainable = False
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model_path = os.path.join("models", model_name + ".hdf5") # -{epoch:04d}-{val_loss:.2f}
    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)

    for epoch in range(Config.TRAIN_EPOCH):
        # train the model on the new data for a few epochs
        data_order = [i for i in range(Config.TRAIN_START, Config.TRAIN_END+1)]
        shuffle(data_order)
        for count in data_order:
            data_filename = get_training_name(count)
            print(data_filename)
            data = np.load(data_filename)

            X = np.array([i[0] for i in data])
            X = inception_v3.preprocess_input(X)

            Y = np.array([i[1] for i in data])
            print(X.shape)
            print(Y.shape)

            # tensorboard = TensorBoard('./logs', update_freq=1)
            model.fit(X, Y, batch_size=32, epochs=2, verbose=1, validation_split=.2, shuffle=True, callbacks=[checkpoint])



if __name__ == "__main__":
    train_top(model_name="gtav1")
