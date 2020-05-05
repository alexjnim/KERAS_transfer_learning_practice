# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
import math
import os
import gc

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# # LOADING DATA

# +
foldernames = os.listdir('data/oregon_wildlife/')
categories = []
files = []

for k, folder in enumerate(foldernames):
    filenames = os.listdir("data/oregon_wildlife/" + folder);
    for file in filenames:
        files.append("data/oregon_wildlife/" + folder + "/" + file)
        categories.append(folder)
        
df = pd.DataFrame({
    'filename': files,
    'category': categories})

# +
from sklearn.model_selection import train_test_split

train_dataframe, val_dataframe = train_test_split(df, test_size=0.33, random_state=42, stratify=df['category'])

# +
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 64

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
validation_generator = val_datagen.flow_from_dataframe(dataframe=val_dataframe,
                                                    directory = None,
                                                    x_col='filename',
                                                    y_col='category',
                                                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=False,
                                                    class_mode='categorical')
# -

# # PREPPING MODEL PREDICTIONS

ground_truth = validation_generator.classes

# +
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('models/model.h5')

predictions = model.predict_generator(validation_generator)
# -

# make a dictionary storing the image index to the prediction and ground truth

prediction_index = []
for prediction in predictions:
    prediction_index.append(np.argmax(prediction))


def accuracy(predictions, ground_truth):
    total = 0
    for i, j in zip(predictions, ground_truth):
        if i == j:
            total += 1
    return total * 1.0 / len(predictions)


print(accuracy(prediction_index, ground_truth))


# # ANALYSIS FUNCTIONS

def get_images_with_sorted_probabilities(prediction_table,
                                         get_highest_probability,
                                         label,
                                         number_of_items,
                                         only_false_predictions=False):
    sorted_prediction_table = [(k, prediction_table[k])
                               for k in sorted(prediction_table,
                                               key=prediction_table.get,
                                               reverse=get_highest_probability)
                               ]
    result = []
    for index, key in enumerate(sorted_prediction_table):
        image_index, [probability, predicted_index, gt] = key
        if predicted_index == label:
            if only_false_predictions == True:
                if predicted_index != gt:
                    result.append(
                        [image_index, [probability, predicted_index, gt]])
            else:
                result.append(
                    [image_index, [probability, predicted_index, gt]])
    return result[:number_of_items]


# +
def plot_images(filenames, distances, message):
    images = []
    for filename in filenames:
        images.append(mpimg.imread(filename))
    plt.figure(figsize=(20, 15))
    columns = 5
    for i, image in enumerate(images):
        ax = plt.subplot(len(images) / columns + 1, columns, i + 1)
        ax.set_title("\n\n" + filenames[i].split("/")[-1] + "\n" +
                     "\nProbability: " +
                     str(float("{0:.2f}".format(distances[i]))))
        plt.suptitle(message, fontsize=20, fontweight='bold')
        plt.axis('off')
        plt.imshow(image)
        
def display(sorted_indices, message):
    similar_image_paths = []
    distances = []
    for name, value in sorted_indices:
        [probability, predicted_index, gt] = value
#        similar_image_paths.append(VALIDATION_DATA_DIR + filenames[name])
        similar_image_paths.append(df.loc[name]['filename'])
        distances.append(probability)
    plot_images(similar_image_paths, distances, message)


# -

# # ANALYSIS

# prediction_table is a dict with index, prediction, ground truth
prediction_table = {}
for index, val in enumerate(predictions):
    index_of_highest_probability = np.argmax(val)
    value_of_highest_probability = val[index_of_highest_probability]
    prediction_table[index] = [value_of_highest_probability,
index_of_highest_probability, ground_truth[index]]
assert len(predictions) == len(ground_truth) == len(prediction_table)

indices = get_images_with_sorted_probabilities(prediction_table,
                                get_highest_probability=True, label=12, number_of_items=10,
                                only_false_predictions=False)
message = 'Images with the highest probability of containing '
display(indices[:10], message)

indices = get_images_with_sorted_probabilities(prediction_table,
get_highest_probability=False, label=1, number_of_items=10,
only_false_predictions=False)
message = 'Images with the lowest probability of containing '
display(indices[:10], message)

# Incorrect predictions of 'cat'
indices = get_images_with_sorted_probabilities(prediction_table,
get_highest_probability=True, label=0, number_of_items=10,
only_false_predictions=True)
message = 'Images of  with the highest probability of containing '
display(indices[:10], message)


