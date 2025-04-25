#!/usr/bin/env python3.10

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
##                                                                                          ###
##                                 DATASET TO USE                                           ###
##                                                                                          ###
## https://www.kaggle.com/datasets/khlaifiabilel/military-aircraft-recognition-dataset/code ###
##                                                                                          ###
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# Required packages
# import pandas as pd
# import numpy as np
import os
# import xml.etree.ElementTree as ET
# from colorama import Fore, Back, Style
# import random
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import itertools
# from sklearn.preprocessing import OneHotEncoder
import keras_tuner as kt
# from PIL import Image, ImageDraw
import utils as ut
import model as md


# Define the path to the data directory
data_dir = '/home/adeldsk/codes/inteligentni_sistemi/'
# data_set = 'develop'
data_set = 'train'

data = ut.load_train_data(data_dir, data_set)
normal_images, encoded_labels, encoded_labels_np = ut.preprocess_data(data)

n_labels = encoded_labels.shape[1]
build_model_func = md.build_model_n_label(n_labels)

tuner = kt.Hyperband(
    build_model_func,
    objective='val_accuracy',
    max_epochs=100,
    directory='trained_models',
    project_name='IS_CNN_Classificator'
)

x_train, x_test, y_train, y_test = train_test_split(
    normal_images, encoded_labels, test_size=33, random_state=42
)

tuner.search(x_train, y_train, epochs=60, validation_data=(x_test, y_test))

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

printf(f"Best Hyperparameters: {best_hps}")

model = tuner.hypermodel.build(best_hps)
history=model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))

# fix, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(10, 3))
# ax0.imshow(normal_images[0], cmap='gray')
# ax1.imshow(normal_images[30], cmap='gray')
# ax2.imshow(normal_images[20], cmap='gray')
#
# plt.show()

# image = data[0][0]
# bbox = data[0][1]
# draw = ImageDraw.Draw(image)
# for obj in bbox:
#     name = obj.name
#     bounding_box = obj.bbox
#     xmin, ymin, xmax, ymax = bounding_box.xmin, bounding_box.ymin, bounding_box.xmax, bounding_box.ymax
#     draw.rectangle([xmin, ymin, xmax, ymax], outline='red')
#     draw.text((xmin, ymin), name, fill='red')
#
# image.show()
