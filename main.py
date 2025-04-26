#!/usr/bin/env python3.10

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
##                                                                                          ###
##                                 DATASET TO USE                                           ###
##                                                                                          ###
## https://www.kaggle.com/datasets/khlaifiabilel/military-aircraft-recognition-dataset/code ###
##                                                                                          ###
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# Required packages
import os
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras_tuner as kt
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

print(f"Best Hyperparameters: {best_hps}")

model = tuner.hypermodel.build(best_hps)
history=model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))
