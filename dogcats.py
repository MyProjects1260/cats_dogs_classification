# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 22:03:06 2021

@author: Ben Romdhane
"""
import tensorflow as tf
import numpy as np

import glob
pathDog = glob.glob("C:/Users/Ben Romdhane/Downloads/Dogs.Cats/Data/dog*")
pathCat = glob.glob("C:/Users/Ben Romdhane/Downloads/Dogs.Cats/Data/cat*")
path = pathDog + pathCat
print(path)

from PIL import Image
img=Image.open("C:/Users/Ben Romdhane/Downloads/Dogs.Cats/Data/dog.4047.jpg")
img

from PIL import Image
import numpy as np
Data=[]
for img in path:

   img1=Image.open(img)
   imgResize = img1.resize((128,128))
   img2 = np.array(imgResize)
   Data.append(img2)

Data1=np.array(Data)
y=np.repeat(range(1,3),len(pathDog))
print(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(Data1, y, test_size=0.2, random_state=0)


from tensorflow.keras.utils import to_categorical


y_train1=to_categorical(y_train)
y_test1=to_categorical(y_test)

y_train2=y_train1[:,1:]
y_test2=y_test1[:,1:]

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten, Activation
from keras.layers import Dense

classifier = Sequential()
classifier.add(Convolution2D(filters = 32,kernel_size = 9,input_shape = (128,128,3),activation = 'relu'))
classifier.add(MaxPooling2D(7,7))
classifier.add(Flatten()) 

classifier.add(Dense(activation= 'relu',units=100))  
classifier.add(Dense(activation= 'sigmoid',units=2)) 

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
history = classifier.fit(X_train, y_train2, batch_size=32, epochs=10, validation_data=(X_test, y_test2))


score = classifier.evaluate(X_test , y_test2)
print("test loss", score[0])
print("test accuracy", score[1])

from matplotlib import pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Training and validation accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train_acc', 'val_acc'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train_loss', 'val_loss'])
plt.show()
