# -*- coding: utf-8 -*-
"""
Created on Tue May 26 10:27:43 2020

@author: hanna
"""


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np

import matplotlib.pyplot as plt
from corpus_v5 import *
from sklearn.model_selection import train_test_split


def gen_data(n = 5, N = 100, test_size = 0.1, val_test_size = 0.5):
    corpus = Corpus_n(n = n, target = 1, initial = 0)
    corpus.generate_random_graphs(N=N, total_random = True, verbose = False)
    print('-'*10 + ' Corpus done! ' + '-'*10)
    
    data_X = np.ones((N, n, n))
    data_labels = np.ones((N,2))
    for i in range(N): 
        x = corpus.corpus_list[i].A
        data_X[i] = x # numpy array
        data_labels[i] = corpus.corpus_list[i].label # 2 dim np array, categorical
    

    data_X = data_X.reshape(N, n, n, 1) # [samples, rows, columns, channels]
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_labels, test_size=test_size)

    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=val_test_size)
    np.savez('train_val_test_data', X_train, y_train, X_test, y_test, X_val, y_val)

def load_data(filename):
    file = np.load(filename)
    X_train = file['arr_0']
    y_train = file['arr_1']
    X_test = file['arr_2']
    y_test = file['arr_3']
    X_val = file['arr_4']
    y_val = file['arr_5']
    return X_train, y_train, X_test, y_test, X_val, y_val
        
# n is a global! FIX This!
def kernel_init(shape, dtype=None): 
    z = np.zeros((n-1, n-1))
    o = np.ones((n-1,1))
    detector1 = np.concatenate((z, o, z), axis = 1)
    detector2 = np.ones((1, n*2-1))
    detector2[0,n-1] = 0
    detector = np.concatenate((detector1, detector2, detector1), axis = 0)
    weights = detector.reshape(shape)
    
    return tf.convert_to_tensor(weights, dtype=dtype)


class trainable_ETE_ETV_Net(tf.keras.Model):
    
    def __init__(self, n):
        super(trainable_ETE_ETV_Net, self).__init__()
        self.n = n
    
    def build(self):
        inputs = tf.keras.Input(shape=(n,n,1), batch_size = batch_size)
        #inputs = tf.keras.Input(shape=(batch_size,n,n,1))
        
        # ETE
        x = layers.ZeroPadding2D(padding=(n-1, n-1))(inputs)
        x = layers.Conv2D(1, (2*n-1,2*n-1), kernel_initializer = kernel_init)(x)
        x = tf.math.multiply(x, inputs)
        
        # ETV
        y = layers.ZeroPadding2D(padding=(n-1, n-1))(inputs)
        y = layers.Conv2D(1, (2*n-1,2*n-1), kernel_initializer = kernel_init)(y)
        y = tf.linalg.diag_part(y)
        y = tf.reshape(y, (batch_size, n, 1, 1))
        
        combined = tf.concat([x, y], axis = 2)
        
        z = tf.keras.layers.Conv2D(n, (3,3))(combined)
        
        # Fully connected
        z = tf.keras.layers.Flatten()(z)
        z = tf.keras.layers.Dense(3*n, activation='relu')(z) # according to Melnikov 2020
        z = tf.keras.layers.Dense(10, activation='relu')(z) # according to Melnikov 2020
        z = tf.keras.layers.Dense(2)(z)
        model = tf.keras.models.Model(inputs, z)
        
        model.summary()
    
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        return model

batch_size = 10 # has to be a multiple of how many train and test samples
verbose = True
epochs = 200
N=1000 # number of random generated graphs
n = 5
test_size = 0.1
val_test_size = 0.5
gen_data(n = n, N = N, test_size = test_size, val_test_size = val_test_size)
X_train, y_train, X_test, y_test, X_val, y_val = load_data('train_val_test_data.npz')

model = trainable_ETE_ETV_Net(n)
model = model.build()

history = model.fit(X_train, y_train, batch_size=batch_size, validation_data = (X_val, y_val), epochs=epochs, verbose=verbose)

test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)


predictions = model.predict(X_test)

if verbose:
    plt.ylim(0.0, 1.0)
    plt.plot(np.linspace(0.0, epochs, epochs), history.history['loss'],'--', label = 'loss for ' + str(n) + ' nodes')
    plt.plot(np.linspace(0.0, epochs, epochs), history.history['accuracy'],'-', label = 'accuracy for ' + str(n) + ' nodes')
    plt.xlabel('epochs')
    plt.ylabel('learning performance')
    plt.legend()