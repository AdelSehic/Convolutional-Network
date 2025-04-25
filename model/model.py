# Required packages
# import pandas as pd
# import numpy as np
import os
# import xml.etree.ElementTree as ET
# from colorama import Fore, Back, Style
# import random
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# import itertools
# from sklearn.preprocessing import OneHotEncoder
import keras_tuner as kt
from PIL import Image, ImageDraw
import utils as ut
from . import config as cfg


def build_model_n_label(n_labels):
    def build_model(hp):
        model = keras.Sequential()
        model.add(keras.Input(shape=cfg.INPUT_SHAPE))
        model.add(layers.Conv2D(cfg.KERNEL_N, cfg.KERNEL_SIZE, activation=cfg.ACTIVATION))
        model.add(layers.AveragePooling2D(cfg.POOL_SIZE))

        for i in range(hp.Int('conv_layers', cfg.MIN_CONV_LAYERS, cfg.MAX_CONV_LAYERS)):
            filters = hp.Int(f'filters_{i}', cfg.MIN_FILTERS, cfg.MAX_FILTERS, step=cfg.FILTER_STEP)
            model.add(layers.Conv2D(filters, cfg.KERNEL_SIZE, activation=cfg.ACTIVATION))
            model.add(layers.AveragePooling2D(cfg.POOL_SIZE))

        model.add(layers.Flatten())
        model.add(layers.Dense(hp.Int('units', cfg.MIN_FILTERS_DENSE, cfg.MAX_FILTERS_DENSE, cfg.FILTER_STEP), activation=cfg.ACTIVATION))
        model.add(layers.Dropout(hp.Float('dropout', cfg.MIN_DROPOUT, cfg.MAX_DROPOUT, cfg.DROPOUT_STEP)))

        hp_learning_rate = hp.Choice('learning_rate', values=cfg.LEARNING_RATE)
        optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate)

        model.add(layers.Dense(n_labels, activation='softmax'))

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        return model
    return build_model
