from __future__ import absolute_import, division, print_function, unicode_literals
import neptune
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D,GlobalAveragePooling2D, Concatenate, Reshape,GlobalMaxPooling2D, Activation, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from classification_models.tfkeras import Classifiers
from PIL import Image
import numpy as np
import pandas
import os
import pathlib
import datetime
import math
import sys
import optuna

# GPU setup
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True) 

# Config loading

train_path = "../../bachelor-data/data_resize/allTrain.csv"
validate_path ="../../bachelor-data/data_resize/allTest.csv"

image_dir = "../../bachelor-data/data_resize/"
checkpointpath = "../../bachelor-data/checkpoints/"
modelName = sys.argv[0]

learning_rate = 0.001

image_height = 224
image_width = 224
batch_size = 32
numEpochs = 500

conf= {
        "train_path": train_path,
        "validate_path": validate_path,
        "image_dir": image_dir,
        "modelName": modelName,
        "learning_rate": learning_rate,
        "image_height": image_height,
        "image_width": image_width,
        "batch_size": batch_size,
        "numEpochs": numEpochs
        }




# Model
def objective(trial):

    ResNet18, preprocess_input = Classifiers.get('resnet18')
    RESNET = ResNet18(include_top=False, weights='imagenet', input_shape=(image_height,image_width,3))
    model = tf.keras.Sequential()

    
    # Projection
    doProjection = trial.suggest_categorical('projection', ['yes','no'])
    if doProjection == 'yes':
        model.add(Conv2D(3,(1,1),input_shape=(image_height,image_width,1),padding="same"))


    dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.5)
    # Resnet
    model.add(RESNET)

    model.add(GlobalAveragePooling2D())

    model.add(Dense(trial.suggest_int("num_neurons_1",1,512),Activation("relu")))
    model.add(Dropout(dropout_rate))
    model.add(Dense(trial.suggest_int("num_neurons_2",1,512),Activation("relu")))
    model.add(Dropout(dropout_rate))

    model.add(Dense(1))


    optimize = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimize,
                  loss='MSE',
                  metrics=['mse']
                  )



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

    if model.input_shape[2] == 1:

        train_generator = train_datagen.flow_from_dataframe(
                dataframe=train_df,
                directory=image_dir,
                x_col="filename",
                y_col='label',
                target_size=(image_height, image_width),
                batch_size=batch_size,
                shuffle=True,
                class_mode="raw",
                color_mode="grayscale"
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
                color_mode="grayscale"
                )
    else:

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


    filepath=str(checkpointpath)+"model_"+str(modelName)+"_checkpoint-"+str(image_height)+"x"+str(image_width)+"-{epoch:03d}-{val_mse:.16f}.hdf5"

    RLR = keras.callbacks.ReduceLROnPlateau(monitor='val_mse', factor=0.5, patience=2, verbose=1, mode='min', min_delta=0.0001, cooldown=0)

    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_mse', verbose=0, save_best_only=True, save_weights_only=False, mode='min')

    earlyStop = keras.callbacks.EarlyStopping(monitor='val_mse', mode='min', patience=10, restore_best_weights=True,verbose=1)



    callbacks_list = [checkpoint, RLR, earlyStop]

    model.summary()
    history = model.fit(train_generator,validation_data=val_generator,verbose=1 , epochs=numEpochs, steps_per_epoch=train_generator.n/train_generator.batch_size , callbacks=callbacks_list)

    

    val = history.history['val_mse']
    os.system("rm /home/lasg/bachelor-data/checkpoints/*")
    return val[-1]

study = optuna.create_study(direction='minimize', storage='sqlite:///optunaDB.db')
study.optimize(objective, n_trials=100)
