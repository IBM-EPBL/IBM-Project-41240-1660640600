#!/usr/bin/env python
# coding: utf-8

# In[60]:


import numpy as np #used for numerical analysis
import tensorflow #open source used for both ML and DL for computatio
from tensorflow.keras.models import Sequential #it is a palin stack of layer 
from tensorflow.keras.layers import Layer #Alayer consists of a tensor-in tensor-out computation function
from tensorflow.keras.layers import Dense,Flatten #flattening the input or change the dimension
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout #convolution layer , MaxPooling2D-for downsampling the image
from keras.preprocessing.image import ImageDataGenerator


# In[61]:


model=Sequential()


# In[62]:


model.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))


# In[63]:


model.add(MaxPooling2D(pool_size =(2,2)))


# In[64]:


model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size =(2,2)))
model.add(Flatten())


# In[65]:


model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=128,activation='softmax'))


# In[66]:


model.summary


# In[ ]:




