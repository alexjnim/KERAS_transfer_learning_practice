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

# +
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.inception_v3 import InceptionV3

def model_picker(name, IMG_WIDTH, IMG_HEIGHT):
    if (name == 'vgg16'):
        model = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
                 #     pooling='max')
                     )
    elif (name == 'vgg19'):
        model = VGG19(weights='imagenet',
                      include_top=False,
                      input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
#                      pooling='max')
                     )
    elif (name == 'mobilenet'):
        model = MobileNet(weights='imagenet',
                          include_top=False,
                          input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
                          depth_multiplier=1,
                          alpha=1)
    elif (name == 'inception'):
        model = InceptionV3(weights='imagenet',
                            include_top=False,
                            input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
 #                           pooling='max')
                           )
    elif (name == 'resnet'):
        model = ResNet50(weights='imagenet',
                         include_top=False,
                         input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
#                        pooling='max')
                        )
    elif (name == 'xception'):
        model = Xception(weights='imagenet',
                         include_top=False,
                         input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
#                         pooling='max')
                        )
    else:
        print("Specified model not available")
    return model


# -

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
# -

df.head()

# # AUGMENTING DATA
#
# Here we use the Keras ImageDataGenerator to augment the data

# +
from sklearn.model_selection import train_test_split

train_dataframe, val_dataframe = train_test_split(df, test_size=0.33, random_state=42, stratify=df['category'])
# -

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.2,
                                   validation_split = 0.3)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# note: you can also do this with flow_from_directory so that extracting path names into a dataframe is not necessary

# +
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 64

train_generator = train_datagen.flow_from_dataframe(dataframe=train_dataframe,
                                                    directory = None,
                                                    x_col='filename',
                                                    y_col='category',
                                                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    seed=42,
                                                    class_mode='categorical')
validation_generator = val_datagen.flow_from_dataframe(dataframe=val_dataframe,
                                                    directory = None,
                                                    x_col='filename',
                                                    y_col='category',
                                                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=False,
                                                    class_mode='categorical')


# -

# # DEFINE THE MODEL

def model_maker(MODEL_NAME, IMG_WIDTH, IMG_HEIGHT):
    from keras.applications.vgg16 import VGG16
#    base_model = MobileNet(include_top=False,
 #                          input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
#    base_model = VGG16(include_top=False,
#                      input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
#                        weights='imagenet',
#                        classes = 20)

    base_model = model_picker(MODEL_NAME, IMG_WIDTH, IMG_HEIGHT)
    for layer in base_model.layers[:]:
        layer.trainable = False

    input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    custom_model = base_model(input)
    custom_model = GlobalAveragePooling2D()(custom_model)
    custom_model = Dense(64, activation='relu')(custom_model)
    custom_model = Dropout(0.5)(custom_model)
    predictions = Dense(NUM_CLASSES, activation='softmax')(custom_model)
    return Model(inputs=input, outputs=predictions)


TRAIN_SAMPLES = 500
VALIDATION_SAMPLES = 500
NUM_CLASSES = len(df['category'].unique())

# +
MODEL_NAME = 'vgg16'

model = model_maker(MODEL_NAME, IMG_WIDTH, IMG_HEIGHT)

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(0.001),
              metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=math.ceil(float(TRAIN_SAMPLES) / BATCH_SIZE),
    epochs=10,
    validation_data=validation_generator,
    validation_steps=math.ceil(float(VALIDATION_SAMPLES) / BATCH_SIZE))
# -

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 3.5)
plt.show()

model.save('models/'+MODEL_NAME+'_model.h5')

# # MODEL PREDICITON

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
model = load_model('models/'+MODEL_NAME+'_model.h5')

img_path = 'data/oregon_wildlife/bald_eagle/0a6bf3fa0a0d17aed4.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = expanded_img_array / 255.  # Preprocess the image
prediction = model.predict(preprocessed_img)
print(prediction)
print(validation_generator.class_indices)




