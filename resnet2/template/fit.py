from __future__ import absolute_import, division, print_function, unicode_literals
import neptune
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import Callback
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

# GPU setup
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True) 

# Config loading
stream = file('config.yml', 'r')    # 'document.yaml' contains a single YAML document.
conf = yaml.safe_load(stream)


train_path = conf['data'][0]['train_path']
validate_path = conf['data'][1]['validate_path']

image_dir = conf['data'][2]['image_dir']
logFileName = conf['data'][3]['log_file']
checkpointpath = conf['data'][4]['checkpoint_path']
modelName = os.getcwd()
modelName = os.path.basename(modelName)

learning_rate = conf['model'][0]['learning_rate']

image_height = conf['image_height']
image_width = conf['image_width']
batch_size = conf['batch_size']
numEpochs = conf['epochs']

# select project
neptune.init('lassegoransson/xrayPredictor')

# Data generators
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
        class_mode="raw",
        color_mode="rgb"
        )

val_generator = val_datagen.flow_from_dataframe(
        dataframe=validate_df,
        directory=image_dir,
        x_col="filename",
        y_col='label',
        target_size=(image_height, image_width),
        batch_size=batch_size,
        shuffle=True,
        class_mode="raw",
        color_mode="rgb"
        )


# Model

RESNET = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(image_height,image_width,3), pooling="avg")

model = tf.keras.Sequential()

#for layer in RESNET.layers:
#  model.add(layer)

#for l in model.layers:
#    l.trainable=False

# Projection
#model.add(Conv2D(3,(3,3),input_shape=(image_height,image_width,1),padding="same"))

model.add(RESNET)
#model.layers[1].trainable=True

model.add(Dense(2048,Activation("relu")))
#model.add(Dropout(0.50))
#model.add(Dense(256,Activation("relu")))
#model.add(Dropout(0.50))
#model.add(Dense(128,Activation("relu")))
#model.add(Dropout(0.50))
#model.add(Dense(64,Activation("relu")))
#model.add(Dropout(0.50))
model.add(Dense(1))


optimize = keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimize,
              loss='MSE',
              metrics=['mse']
              )


class NeptuneMonitor(Callback):
    def on_epoch_end(self, epoch, logs={}):
        neptune.send_metric('val_loss', epoch, logs['val_loss'])
        neptune.send_metric('loss', epoch, logs['loss'])
        neptune.send_metric('learning_rate', epoch, float(tf.keras.backend.get_value(self.model.optimizer.lr)))



filepath=str(checkpointpath)+"model_"+str(modelName)+"_checkpoint-"+str(image_height)+"x"+str(image_width)+"-{epoch:03d}-{val_loss:.12f}.hdf5"

RLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, mode='min', min_delta=0.0001, cooldown=0)

checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min')

earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True,verbose=1)

csvLog = keras.callbacks.CSVLogger(logFileName, separator=str(u','), append=False)

with neptune.create_experiment(name=modelName, params=conf) as npexp:
    neptune_monitor = NeptuneMonitor()

    callbacks_list = [checkpoint,csvLog, neptune_monitor, RLR, earlyStop]

    model.summary()
    model.fit(train_generator,validation_data=val_generator,verbose=1 , epochs=numEpochs, steps_per_epoch=train_generator.n/train_generator.batch_size , callbacks=callbacks_list)

    import glob

    list_of_files = glob.glob(checkpointpath+"*") # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    modelfileName = latest_file 

    npexp.send_artifact(modelfileName)
    tmp = modelfileName.split('-')[3].split('.')
    val = float(tmp[0]+"."+tmp[1])
    neptune.send_metric('val_loss', val)
