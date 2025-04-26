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
import numpy as np


# Define the path to the data directory
data_dir = '/home/adeldsk/codes/inteligentni_sistemi/'
# data_set = 'develop'
data_set = 'train'
test_set = 'test'

data = ut.load_train_data(data_dir, data_set)
normal_images, encoded_labels, encoded_labels_np = ut.preprocess_data(data)
del data

ut.translate_labels(encoded_labels_np)

test_data = ut.load_train_data(data_dir, test_set)
nm_test, encd_test, _ = ut.preprocess_data(test_data)
del test_data

n_labels = encoded_labels.shape[1]
build_model_func = md.build_model_n_label(n_labels)

tuner = kt.Hyperband(
    build_model_func,
    objective='val_accuracy',
    max_epochs=30,
    directory='trained_models',
    project_name='IS_CNN_Classificator'
)

# x_train, x_test, y_train, y_test = train_test_split(
#     normal_images, encoded_labels, test_size=0.33, random_state=42
# )

tuner.search(normal_images, encoded_labels, epochs=20, validation_data=(nm_test, encd_test))

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"Best Hyperparameters: {best_hps}")

model = tuner.hypermodel.build(best_hps)
model.save('trained_models/IS_CNN_Classificator/best_model.keras')

history=model.fit(normal_images, encoded_labels, epochs=30, validation_data=(nm_test, encd_test))

## TEST THE MODEL ##
loss, accuracy = model.evaluate(nm_test, encd_test)

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_pred_probs = model.predict(nm_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(encd_test, axis=1)

# Step 2: Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Step 3: Classification report
print("Classification Report:")
print(classification_report(y_true, y_pred))

from tensorflow.keras.utils import plot_model, model_to_dot
plot_model(model,to_file='basic_model.png')
from IPython.display import SVG
SVG(model_to_dot(model).create(prog='dot',format='svg'))

model.summary()
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Plot the performance of the model
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()
