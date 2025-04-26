#!/usr/bin/env python3.10

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
##                                                                                          ###
##                                 DATASET TO USE                                           ###
##                                                                                          ###
## https://www.kaggle.com/datasets/khlaifiabilel/military-aircraft-recognition-dataset/code ###
##                                                                                          ###
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# Required packages
import utils as ut
import numpy as np
import matplotlib.pyplot as plt
import re

# Custom sort function to sort labels by the numeric value in the label string
def extract_number(label):
    match = re.search(r'(\d+)', label)
    return int(match.group(0)) if match else float('inf')


# Define the path to the data directory
data_dir = '/home/adeldsk/codes/inteligentni_sistemi/'
data_set = 'train'

data = ut.load_train_data(data_dir, data_set)
normal_images, _, encoded_labels_np = ut.preprocess_data(data)
del data

labeled_objects = {}

for obj, label in zip(normal_images, encoded_labels_np):
    if label not in labeled_objects.items():
        labeled_objects[label] = obj

sorted_labels = sorted(labeled_objects.keys(), key=extract_number)
num_rows = len(labeled_objects) // 5 + ( len(labeled_objects) % 5 > 0 )
fig, axes = plt.subplots(num_rows, 5, figsize=(15, num_rows * 3))
axes = axes.flatten()

# Plot each image and label
for i, label in enumerate(sorted_labels):
    img = labeled_objects[label]
    ax = axes[i]
    ax.imshow(img)
    ax.set_title(f"Label: {label}")
    ax.axis('off')

# Hide any extra subplots that do not correspond to labels
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
