#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

img_rows, img_cols = 28, 28
num_classes = 10

def prep_data(raw):
    y = raw[:, 0]
    out_y = keras.utils.to_categorical(y, num_classes)
    print(type(out_y))
    x = raw[:,1:]
    #count number of images by counting number of rows
    num_images = print(raw.shape[0])
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    out_x = out_x / 255 
    print(type(out_x))
    return out_x, out_y

fashion_file = "C:/Users/Palak/Desktop/fashion-mnist_train.csv"
fashion_data = np.loadtxt(fashion_file, skiprows = 1, delimiter = ',')
fashion_data = fashion_data.dropna()
x, y = prep_data(fashion_data)

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D, Dropout

batch_size = 16

#define model as sequential so layers of kernals can be added
fashion_model = Sequential()

# apply two layers of convolutions, 1st also defining the image size
fashion_model.add(Conv2D(16, kernel_size = (5,5), activation = 'relu', input_shape = (img_rows, 
                                                                                      img_cols, 1)))
fashion_model.add(Conv2D(16, kernel_size = (5,5), activation = 'relu'))
fashion_model.add(MaxPool2D(pool_size=(2,2)))
fashion_model.add(Dropout(0.25))

fashion_model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
fashion_model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
fashion_model.add(MaxPool2D(pool_size=(2,2)))
fashion_model.add(Dropout(0.25))



#change to 1-D vector
fashion_model.add(Flatten())

#add dense layer 
fashion_model.add(Dense(128, activation = 'relu'))
fashion_model.add(Dense(num_classes, activation = 'softmax'))

#compile(define loss function, optimizer, and the metrics)
fashion_model.compile(loss = keras.losses.categorical_crossentropy, 
                      optimizer = 'adam', metrics = ['accuracy'])

#finally fit the model with the x,y processed data, define batch size, steps and the validation split
fashion_model.fit(x,y, batch_size = batch_size, epochs = 3, validation_split = 0.2)

