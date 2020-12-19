import os
import time
from random import shuffle

import pandas as pd
import tensorflow.compat.v1 as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

import numpy as np
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model

from config import Config
from data import get_training_name
from model import model_inceptionv3

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

def train_top(model_name, load_weights_path=None):
    backbone, model = model_inceptionv3()

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in backbone.layers:
        layer.trainable = True
    # compile the model (should be done *after* setting layers to non-trainable)
    lr_schedule = schedules.ExponentialDecay(initial_learning_rate=Config.LEARNING_RATE, decay_steps=Config.DECAY_STEPS, decay_rate=Config.DECAY_RATE)
    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    plot_model(model, show_shapes=True, to_file='inceptionv3_architecture.png')
    exit()

    if load_weights_path is not None:
        print(f"Loaded weights {load_weights_path}")
        model.load_weights(load_weights_path)

    model_path = os.path.join("models", model_name + "-{epoch:02d}-{val_loss:.3f}.hdf5")

    df = pd.read_csv(Config.TRAINING_CSV_PATH)
    df["label"] = df["label"].apply(lambda x: str(np.argmax(np.array(x[1:-1].split(","), dtype="int"))))
    num_samples = np.bincount(df["label"])
    class_weight = {}
    for i, num in enumerate(num_samples):
        class_weight[i] = 1. / num
    datagen = ImageDataGenerator(preprocessing_function=inception_v3.preprocess_input, validation_split=Config.VALIDATION_RATIO)
    train_generator = datagen.flow_from_dataframe(subset="training", dataframe=df, directory=Config.TRAINING_IMAGES_PATH, x_col="image", y_col="label", class_mode="categorical", batch_size=Config.BATCH_SIZE, shuffle=True)
    validation_generator = datagen.flow_from_dataframe(subset="validation", dataframe=df, directory=Config.TRAINING_IMAGES_PATH, x_col="image", y_col="label", class_mode="categorical", batch_size=Config.BATCH_SIZE, shuffle=False)

    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=False)
    tensorboard = TensorBoard(log_dir=f"./logs/{model_name}-{int(time.time())}", update_freq=1)
    earlystopping = EarlyStopping(monitor='val_loss', patience=20, min_delta=0.01)
    model.fit(train_generator, initial_epoch=Config.INITIAL_EPOCH, steps_per_epoch = train_generator.samples // Config.BATCH_SIZE, validation_data = validation_generator, validation_steps = validation_generator.samples // Config.BATCH_SIZE, epochs=Config.TRAIN_EPOCH, callbacks=[checkpoint, tensorboard, earlystopping])

    # for epoch in range(Config.TRAIN_EPOCH):
    #     # train the model on the new data for a few epochs
    #     data_order = [i for i in range(Config.TRAIN_START, Config.TRAIN_END+1)]
    #     shuffle(data_order)
    #     for count in data_order:
    #         data_filename = get_training_name(count)
    #         print(data_filename)
    #         data = np.load(data_filename)

    #         X = np.array([i[0] for i in data])
    #         X = inception_v3.preprocess_input(X)

    #         Y = np.array([i[1] for i in data])
    #         print(X.shape)
    #         print(Y.shape)

            # model.fit(X, Y, batch_size=32, epochs=1, verbose=1, validation_split=.2, shuffle=True, callbacks=[checkpoint])



if __name__ == "__main__":
    train_top(model_name=Config.MODEL_NAME, load_weights_path=Config.LOAD_WEIGHTS_PATH)
