import numpy as np
import os
from datetime import datetime

from sklearn.model_selection import train_test_split

import tensorflow as tf

from dataset_utils import create_pairs, PairDataGenerator
from utils import compile_and_fit
from model import siameseLeg, siameseNet

from tensorflow import keras


if __name__ == '__main__':
    ROOT_DIR = 'C:/Users/KK/Desktop/Site-similar/dataset'  # To Fix according to your path
    BATCH_SIZE = 30
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    CHANNELS = 3
    neg_factor = 3
    X, y = create_pairs(ROOT_DIR, neg_factor)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

    train_gen = PairDataGenerator(X_train, y_train, IMG_HEIGHT, IMG_WIDTH, CHANNELS, BATCH_SIZE)
    val_gen = PairDataGenerator(X_val, y_val, IMG_HEIGHT, IMG_WIDTH, CHANNELS, BATCH_SIZE)
    print("TRAIN: ", X_train.shape)
    print("VAL: ", X_val.shape)

    tl, cl = np.unique(y_train, return_counts=True)
    print("TRAIN - LABEL: {} NUM: {} LABEL: {} NUM: {}".format(tl[0], cl[0], tl[1], cl[1]))

    weight_for_0 = (1 / cl[0]) * (len(y_train)) / 2.0
    weight_for_1 = (1 / cl[1]) * (len(y_train)) / 2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))

    initial_bias = np.log([cl[1] / cl[0]])
    print("Initial Bias: ".format(initial_bias[0]))

    tl, cl = np.unique(y_val, return_counts=True)
    print("VAL - LABEL: {} NUM: {} LABEL: {} NUM: {}".format(tl[0], cl[0], tl[1], cl[1]))

    siamese_model = tf.keras.models.load_model('./saved_model/20211105/')

    siamese_model.summary()

    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    fit_logdir = "logs/fit/" + date

    

    def get_callbacks(name):
        return [
            tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=20),
            tf.keras.callbacks.TensorBoard(log_dir=name, histogram_freq=1)
        ]

    history = siamese_model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=1,
            #callbacks=get_callbacks(fit_logdir),
            shuffle=False,
            class_weight=class_weight,
            verbose=1)