from __future__ import print_function
import keras
import time
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import sys
from keras.utils import plot_model


sys.argv[0]="train.py"
learning_rate=sys.argv[1]
batch_size = int(sys.argv[2])
inti=sys.argv[3]
save_dir = sys.argv[4]
num_classes = 10
epochs = 120
data_augmentation = True
num_predictions = 20

#save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'trained_model.h5'


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();
        
plot_losses = PlotLosses()

model = Sequential()

#intializers
if(inti=="1"):
    def my_init(shape, dtype=None):
        return K.glorot_normal(shape, dtype=dtype,seed=None)
if(inti=="2"):
    def my_init(shape, dtype=None):
        return K.he_normal(shape, dtype=dtype,seed=None)

#convolutional layer 1    
model.add(Conv2D(64, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#convolutional layer 2
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#convolutional layer 3
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))

#convolutional layer 4
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#hidden layer 1(FC1)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

#hidden layer 2(FC2)
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

#softmax layer
model.add(Dense(num_classes))
model.add(Activation('softmax'))

#Adam optimizer
opt = keras.optimizers.Adam(lr=float(learning_rate), beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

early_stopping =keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
history=model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.1,callbacks=[early_stopping])
hist=model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test,y_test))


los=history.history['loss']
vallos=history.history['val_loss']

f = open('data.txt','w')
for l in los:
    f.write(str(l))
    f.write(" ")
f.write("val")
for v in vallos:
    f.write(str(v))
    f.write(" ")
f.close()
scores = model.evaluate(x_test,y_test)
model.save(os.path.join(save_dir,'modd.hdf5'))
