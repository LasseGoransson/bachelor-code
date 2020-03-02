from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D,GlobalAveragePooling2D, Concatenate, Reshape,GlobalMaxPooling2D, Activation, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import pandas
import os
import pathlib
import datetime
import math
import sys
import yaml


gpus = tf.config.experimental.list_physical_devices('GPU')


stream = file('config.yml', 'r')    # 'document.yaml' contains a single YAML document.
conf = yaml.safe_load(stream)


train_path = conf['data'][0]['train_path']
validate_path = conf['data'][1]['validate_path']

image_dir = conf['data'][2]['image_dir']
logFileName = conf['data'][3]['log_file']
checkpointpath = conf['data'][4]['checkpoint_path']
modelName = conf['modelname']

learning_rate = conf['model'][0]['learning_rate']

image_height = conf['image_height']
image_width = conf['image_width']
batch_size = conf['batch_size']

train_df = pandas.read_csv(train_path)
validate_df = pandas.read_csv(validate_path)

train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True
        )

val_datagen = ImageDataGenerator(
        rescale=1./255,
        )


train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=image_dir,
        x_col="filename",
        y_col='label',
        target_size=(image_height, image_width),
        batch_size=batch_size,
        shuffle=True,
        color_mode='grayscale',
        class_mode="raw"
        )

val_generator = val_datagen.flow_from_dataframe(
        dataframe=validate_df,
        directory=image_dir,
        x_col="filename",
        y_col='label',
        target_size=(image_height, image_width),
        batch_size=batch_size,
        shuffle=True,
        color_mode='grayscale',
        class_mode="raw"
        )

RESNET = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(image_height,image_width,3), pooling="avg")

model = tf.keras.Sequential()

#for layer in RESNET.layers:
#  model.add(layer)

#for l in model.layers:
#    l.trainable=False

model.add(Conv2D(3,(1,1),input_shape=(image_height,image_width,1),name='main_input'))

model.add(RESNET)
model.layers[1].trainable=False

model.add(Dense(512,Activation("relu")))
model.add(Dense(256,Activation("relu")))
model.add(Dense(1))


optimize = keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimize,
              loss='MSE',
              metrics=['mse']
              )

currentEpoch=0

filepath=str(checkpointpath)+"model_"+str(modelName)+"_checkpoint-"+str(image_height)+"x"+str(image_width)+"-{epoch:03d}-{val_loss:.5f}.hdf5"

checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min')

earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=25, restore_best_weights=True,verbose=1)

csvLog = keras.callbacks.CSVLogger(logFileName, separator=str(u','), append=True)

callbacks_list = [checkpoint,csvLog]

model.summary()
history = model.fit(train_generator,validation_data=val_generator,verbose=1 , epochs=currentEpoch+(50-currentEpoch), steps_per_epoch=train_generator.n/train_generator.batch_size ,initial_epoch=currentEpoch, callbacks=callbacks_list)

currentEpoch=len(history.history["loss"])
