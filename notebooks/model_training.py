import pandas as pd
import numpy as np
import tensorflow as tf
import os
import glob
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense , Input


train_dir = 'data/train'
test_dir = 'data/test'

train_real = glob.glob("data/train/REAL/*.jpg", recursive=True)
train_fake = glob.glob("data/train/FAKE/*.jpg", recursive=True)
test_real = glob.glob("data/test/REAL/*.jpg", recursive=True)
test_fake = glob.glob("data/test/FAKE/*.jpg", recursive=True)

print(len(train_real), len(train_fake), len(test_real), len(test_fake))


def load_images(image_paths):
    images = []
    for path in image_paths:
        img = image.load_img(path, target_size=(32, 32))
        img_array = image.img_to_array(img)
        images.append(img_array)
    return np.array(images)

# لود تصاویر
X_train_real = load_images(train_real)
X_train_fake = load_images(train_fake)
X_test_real = load_images(test_real)
X_test_fake = load_images(test_fake)

# ترکیب
X_train = np.concatenate([X_train_real, X_train_fake])
X_test = np.concatenate([X_test_real, X_test_fake])

# labelها
y_train = np.concatenate([np.ones(len(train_real)), np.zeros(len(train_fake))])
y_test = np.concatenate([np.ones(len(test_real)), np.zeros(len(test_fake))])

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


model = Sequential()

model.add(Input(shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu')) 
model.add(MaxPooling2D((2, 2)))                                           

model.add(Conv2D(32, (3, 3), activation='relu'))                         
model.add(MaxPooling2D((2, 2)))


model.add(Flatten())                                                       
model.add(Dense(64, activation='relu'))                                  
model.add(Dense(1, activation='sigmoid'))                                


model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

model.summary()   

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
