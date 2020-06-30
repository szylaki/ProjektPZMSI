# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 13:25:19 2020

@author: FireWalker
"""
from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model

import sklearn.datasets
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import math



(x_train, _), (x_test, _) = mnist.load_data()
of = sklearn.datasets.fetch_olivetti_faces()
X = of["data"]
y = of["target"]
test_size=0.2

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

y_train=y_train.astype('float32')/255.
y_test = y_test.astype('float32') / 255.
print (x_train.shape)
print (x_test.shape)
print(y_train.shape)
print(y_test.shape)


# wielkosc encodera
encoding_dim = 128  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

shape=x_train.shape[1]

# Przykład
input_img = Input(shape=(shape,))
# "encoded" zenkodowana wersja inputu
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" stratna dekompresja img
decoded = Dense(shape, activation='sigmoid')(encoded)

# Model mapuje wejscie dla rekonstrukcji
autoencoder = Model(input_img, decoded)

# ten model mapuje wejscie dla jego zakodowanej wartosci
encoder = Model(input_img, encoded)

# tworzenie przykładu dla zenkodowanego inputu 32
encoded_input = Input(shape=(encoding_dim,))
# odbierz ostatnią warstwę autoencodera
decoder_layer = autoencoder.layers[-1]
# stwórz decoder
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                epochs=20,
                batch_size=1, #256(500) - 0.5802  150(500) - 0.0498
                shuffle=True,
                validation_data=(x_test, x_test))


# zakoduj i odkoduj jakies liczby
# liczby są wzięte ze zbioru x_test
encoded_imgs = encoder.predict(x_test[0])
decoded_imgs = decoder.predict(encoded_imgs)
print("Compression of factor for",encoding_dim,"floats, is:",x_train.shape[1]/encoding_dim)
#======================================================================

n = 10  # Ile liter chcemy wyswietlić
plt.figure(figsize=(20, 4))
for i in range(n):
    # Oryginał
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(int(math.sqrt(shape)), int(math.sqrt(shape))))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # rekonstrukcja
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(int(math.sqrt(shape)), int(math.sqrt(shape))))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()