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
# from sklearn.model_selection import train_test_split
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
data_set = 'develop'
test_set = 'test'

data = ut.load_train_data(data_dir, data_set)
normal_images, encoded_labels, encoded_labels_np = ut.preprocess_data(data)

md.n_labels = encoded_labels.shape[1]

tuner = kt.Hyperband(
    md.build_model,
    objective='val_accuracy',
    max_epochs=100,
    directory='trained_models',
    project_name='IS_CNN_Classificator'
)

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
