import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import (BatchNormalization, Conv2D, Conv2DTranspose, Dense,
                          Dropout, Flatten, Input, Reshape, UpSampling2D,
                          ZeroPadding2D)
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam

np.random.seed(10)

#get file name
model_filename= "Temp"
noise_dim = 100
batch_size = 16
steps_epoch = 3750 
epochs = 50

rows, cols, channels = 28, 28, 1

my_optimizer = Adam(0.0002, 0.5)

# Download Mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalize to between -1 and 1
x_train = (x_train.astype(np.float32) - 127.5) / 127.5

#flatten
x_train = x_train.reshape(-1, rows*cols*channels)

def create_generator():
    generator = Sequential()
    
    generator.add(Dense(256, input_dim=noise_dim))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(rows*cols*channels, activation='tanh')) #output
    
    generator.compile(loss='binary_crossentropy', optimizer=my_optimizer)
    return generator

def create_descriminator():
    discriminator = Sequential()
     
    discriminator.add(Dense(1024, input_dim=rows*cols*channels))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(1, activation='sigmoid'))
    
    discriminator.compile(loss='binary_crossentropy', optimizer=my_optimizer)
    return discriminator

discriminator = create_descriminator()
discriminator.trainable = False

generator = create_generator()

my_gan_input = Input(shape=(noise_dim,))
fake_img = generator(my_gan_input)
my_gan_output = discriminator(fake_img)

my_gan = Model(my_gan_input, my_gan_output)
my_gan.compile(loss='binary_crossentropy', optimizer=optimizer)

for epoch in range(epochs):
    for batch in range(steps_epoch):
        random_noise = np.random.normal(0, 1, size=(batch_size, noise_dim))
        fke_x = generator.predict(random_noise)

        real_x = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]
        
        x = np.concatenate((real_x, fke_x))

        discri_y = np.zeros(2*batch_size)
        discri_y[:batch_size] = 0.9

        d_loss = discriminator.train_on_batch(x, discri_y)

        y_gen = np.ones(batch_size)
        g_loss = my_gan.train_on_batch(random_noise, y_gen)

    print(f'Epoch: {epoch} \t Discriminator Loss: {d_loss} \t\t Generator Loss: {g_loss}')

generator.save(model_filename)
