#!/usr/bin/env python
# coding: utf-8

# In[91]:


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
import matplotlib.pyplot as plt
import os
import pathlib
import datetime
import math
import sys
import neptune

def plotFromCsv(file):

    data = np.genfromtxt(file, delimiter=',', skip_header=1, names=['epoch','loss','mse','val_loss','val_mse'])
    plt.figure(figsize=(20,10))
    plt.plot(data['epoch'],data['val_loss'])
    plt.plot(data['epoch'],data['loss'])
    plt.title('Model accuracy (' +str(file)+")")
    plt.ylabel('Loss (MSE) (KG)')
    plt.xlabel('Epoch')
    plt.legend(['Validation Loss','Loss'], loc='upper right')
    plt.ylim(0,0.8)
    plt.show()


# xrayPredictor
# xrayPredictor-448x448
# xrayPredictor-custom

# In[92]:


name = str(sys.argv[1])
project = 'lassegoransson/xrayPredictor-rerun'
import neptune
key = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiOWUxN2YwMTUtNjM1Ny00NmVlLWIzOTctNzAwYTllMGNmMTg2In0="
project = neptune.init(project,api_token=key)


# In[93]:


get_ipython().system('rm output/*')
ex = exp = project.get_experiments(name)[0]
ex.download_artifacts()
get_ipython().system('unzip output.zip')


# In[94]:


files = get_ipython().getoutput('find output/ -name "model*"')
files


# In[95]:


modelname = files[0]
modelpath = str(modelname.split("/")[1])
print(modelpath)
model = tf.keras.models.load_model(modelname, custom_objects={'Activation': tf.keras.layers.Activation})


# In[ ]:





# In[96]:


image_height = model.input_shape[1]
image_width = model.input_shape[2]
print(image_height)
print(image_width)

img = get_ipython().getoutput('head -n2 ~/bachelor-data/allTest.csv')
img = img[1].split(",")[0]
imgdir=""
imgdir = sys.argv[2]


test_df = pandas.read_csv("/home/lasg/bachelor-data/allTest.csv")
test_datagen = ImageDataGenerator(
        rescale=1./255,
        )

shape = model.input_shape
height = shape[1]
width = shape[2]

if model.input_shape[3] == 1:
    
    test_generator = test_datagen.flow_from_dataframe(
            dataframe=test_df,
            directory=imgdir,
            x_col="filename",
            y_col='label',
            target_size=(image_height, image_width),
            batch_size=1,
            shuffle=False,
            class_mode="raw",
            color_mode="grayscale"
            #color_mode="rgb"
            )
else: 
        test_generator = test_datagen.flow_from_dataframe(
            dataframe=test_df,
            directory=imgdir,
            x_col="filename",
            y_col='label',
            target_size=(image_height, image_width),
            batch_size=1,
            shuffle=False,
            class_mode="raw",
            #color_mode="grayscale"
            color_mode="rgb"
            )
        


# # Evaluate
# 
# 
# 

# In[97]:


n = []
nnon=[]
mse=0
hits=[0,0,0,0,0,0]
labeltrueVal = 0
labelval = 0
i=0
for b in range(0,test_generator.n):
  img,y= test_generator.next()
  img=img[0]
  sys.stdout.write("\r" + str(100*i/test_generator.n))
  label = y[0]
  predict = model.predict(np.expand_dims(img, axis=0))[0][0]
  val =(1-(label/predict))
  mse+=((1/test_generator.n)*(label-predict)*(label-predict))
  nnon.append(val)
  val=abs(val)
  n.append(val)
  if val < 0.05:
    hits[0] += 1
  if val < 0.10:
    hits[1] += 1
  if val < 0.15:
    hits[2] += 1
  if val < 0.20:
    hits[3] += 1
  if val < 0.25:
    hits[4] += 1
  if val < 0.30:
    hits[5] += 1

  #BATCH
  labeltrueVal += label
  labelval +=predict
  i+=1
  


# # Model Architecture

# In[ ]:


modelname.split("/")[1]


# In[ ]:




# # SCORE

# In[ ]:


print("")
print("MSE: "+str(mse))
print("")
print("Deviation from true weight (< 5% = 90 is project goal)")
print("< 5.0% = " +  str(100*hits[0]/(test_generator.n))+ "%")
print("< 10.0% = " + str(100*hits[1]/(test_generator.n))+ "%")
print("< 15.0% = " + str(100*hits[2]/(test_generator.n))+ "%")
print("< 20.0% = " + str(100*hits[3]/(test_generator.n))+ "%")
print("< 25.0% = " + str(100*hits[4]/(test_generator.n))+ "%")
print("< 30.0% = " + str(100*hits[5]/(test_generator.n))+ "%")
print("")
print("Deviation summed over full batch")
print("Predicted weight: "+str(labelval)+" True weight: "+str(labeltrueVal)+" Percentage: "+str((labelval/labeltrueVal)*100))


# In[ ]:


ex.log_metric("5%_score",100*hits[0]/(test_generator.n))
ex.log_metric("10%_score",100*hits[1]/(test_generator.n))
ex.log_metric("15%_score",100*hits[2]/(test_generator.n))
ex.log_metric("20%_score",100*hits[3]/(test_generator.n))
ex.log_metric("25%_score",100*hits[4]/(test_generator.n))
ex.log_metric("30%_score",100*hits[5]/(test_generator.n))


# # Residuals

# In[ ]:


plt.figure(figsize=(20,10))
plt.stem(range(0,len(n)),nnon,use_line_collection=True)
plt.title('Model accuracy')
plt.ylabel('Accuracy (%)')
plt.xlabel('N of test')
plt.show()

