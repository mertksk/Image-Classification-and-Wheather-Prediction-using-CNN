# -- coding: utf-8 --
"""
Created on Fri Jun 10 17:32:09 2022

@author: mert_
"""
import pandas as pd
from PIL import Image
import pydicom as dicom
from skimage import exposure
from skimage.transform import resize
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import seaborn as sns
import keras

 
 
ImagesandLabels = []

rootdir=os.path.dirname(os.path.abspath(__file__))
inputdir = rootdir + r"\Dataset" +"/"

for filename in os.listdir(inputdir):    
    img = Image.open( inputdir + filename)
    img_array = np.array(img)
    img_comp = resize(img_array,(250,250,3))
    
    class_1 = lambda class_dsc,class_id:('sunrise',0) if 'sunrise' in filename else (('shine',1) if 'shine' in filename else ('cloudy',2) )
    
    ImagesandLabels.append({ "img":img_comp,"class_des":class_1(filename,filename)[0],"class_id":class_1(filename,filename)[1]})
    
img_df = pd.DataFrame(ImagesandLabels)
 
np.random.shuffle(ImagesandLabels)
 
Img = []
Label = []
for i,item in enumerate(ImagesandLabels):
    Img.append(item['img'])
    Label.append(item['class_id'])
    
Img = np.array(Img)
Label = np.array(Label)



Img_train = Img[:750]
Label_train = Label[:750]

Img_test = Img[750:]
Label_test = Label[750:]
 


Label_train = keras.utils.to_categorical(Label_train, 3)
Label_test = keras.utils.to_categorical(Label_test, 3)

model = Sequential()


model.add(Conv2D(16, (5, 5), strides = (2,2), padding='same',input_shape=Img_train.shape[1:]))
model.add(Activation('relu'))


model.add(Conv2D(32, (5, 5), strides = (2,2)))
model.add(Activation('relu'))


model.add(Conv2D(64, (5, 5), strides = (2,2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dense(256))
model.add(Activation('relu'))

model.add(Dense(3))
model.add(Activation('softmax'))

batch_size = 64
 
opt = keras.optimizers.RMSprop(lr= 1e-5)
 
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

model_history = model.fit(Img_train, Label_train, batch_size=batch_size, epochs=5, validation_data=(Img_test, Label_test),shuffle=True)
 
fig, ax = plt.subplots()
ax.plot(model_history.history["val_accuracy"],'r', marker='.', label="Model  Accuracy")
ax.plot(model_history.history["val_loss"],'y', marker='.', label="Validation Loss")
ax.plot(model_history.history["loss"],'g', marker='.', label="Train Loss")

ax.set_xlabel('Epoch Number')
ax.set_xlim(0,5)
ax.set_title('Different Datas')
ax.legend(fontsize='small',loc=(0,1))