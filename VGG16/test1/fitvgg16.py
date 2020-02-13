from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D,GlobalAveragePooling2D, Concatenate, Reshape,GlobalMaxPooling2D, Activation, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import IPython.display as display
from PIL import Image
import numpy as np
import pandas
import os
import pathlib
import datetime
import math
import sys
import yaml

modelNum = 3
BS = 8 # Batch size
image_height = 1280
image_width = 2000

train_df = pandas.read_csv("data/allTrain.csv")
validate_df = pandas.read_csv("data/allVal.csv")
test_df = pandas.read_csv("data/allTest.csv")

train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True
        )

val_datagen = ImageDataGenerator(
        rescale=1./255,
        )


test_datagen = ImageDataGenerator(
        rescale=1./255,
        )

train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory='',
        x_col="filename",
        y_col='label',
        target_size=(image_height, image_width),
        batch_size=BS,
        shuffle=True,
        class_mode="raw"
        )

val_generator = val_datagen.flow_from_dataframe(
        dataframe=validate_df,
        directory='',
        x_col="filename",
        y_col='label',
        target_size=(image_height, image_width),
        batch_size=BS,
        shuffle=True,
        class_mode="raw"
        )

test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory='',
        x_col="filename",
        y_col='label',
        target_size=(image_height, image_width),
        batch_size=1,
        shuffle=False,
        class_mode="raw"
        )


VGG16_MODEL = tf.keras.applications.VGG16(input_shape=(image_height,image_width,3),include_top=False,weights="imagenet")

model = tf.keras.Sequential()

for layer in VGG16_MODEL.layers:
  model.add(layer)

for l in model.layers:
    l.trainable=False

model.add(GlobalAveragePooling2D())
#model.add(Flatten())

#model.add(Dense(4096))

model.add(Dense(512))

model.add(Dense(1))
model.summary()


optimize = keras.optimizers.RMSprop(learning_rate=0.0001)
model.compile(optimizer=optimize,
              loss='MSE',
              metrics=['mse']
              )

currentEpoch=0

#!mkdir "checkpoints"
filepath="pythonscripts/test2/checkpoints/model_"+str(modelNum)+"_checkpoint-"+str(image_height)+"x"+str(image_width)+"-{epoch:03d}-{val_loss:.5f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min')
earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=25, restore_best_weights=True,verbose=1)
callbacks_list = [checkpoint]
history = model.fit(train_generator,validation_data=val_generator,verbose=1 , epochs=currentEpoch+(300-currentEpoch), steps_per_epoch=train_generator.n/train_generator.batch_size ,initial_epoch=currentEpoch, callbacks=callbacks_list)
currentEpoch=len(history.history["loss"])
