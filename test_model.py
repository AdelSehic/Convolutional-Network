#!/usr/bin/env python3.10

import tensorflow as tf
from tensorflow.keras.utils import plot_model, model_to_dot
import utils as ut

model = tf.keras.models.load_model('trained_models/IS_CNN_Classificator/best_model.keras')

data_dir = '/home/adeldsk/codes/inteligentni_sistemi/'
test_data = ut.load_train_data(data_dir, 'test')
normal_test_images, encoded_test_labels, _ = ut.preprocess_data(test_data)

loss, accuracy = model.evaluate(normal_test_images, encoded_test_labels)
print(f"Test accuracy: {accuracy:.4f}")
