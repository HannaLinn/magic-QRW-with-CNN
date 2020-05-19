# -*- coding: utf-8 -*-
"""
Created on Thu May 14 11:58:06 2020

@author: hanna
"""

'''
TO DO:
1. Wrong loss function, fixed (?), use from_logits=True?
2. Add convolutional filters
3. single batch of 3 examples per epoch, translated to batch size of 3
4. add conv layers
'''

import tensorflow as tf
from tensorflow.keras import layers, models

import numpy as np
import matplotlib.pyplot as plt
from corpus_v4 import *
from sklearn.model_selection import train_test_split

verbose = True

largest_n = 6 # n is number if nodes in the graphs
smallest_n = 5
epochs = 3
N=10 # number of random generated graphs


def edge_to_edge(input):
    y = tf.numpy_function(np_edge_to_edge, [input], tf.float32)
    return y

def edge_to_vertex(input):
    y = tf.numpy_function(np_edge_to_vertex, [input], tf.float32)
    return y

def np_edge_to_edge(A):
    n = len(A)
    summa = np.zeros((n,n))
    for j in range(n):
        for i in range(n):
            for k in range(n):
                summa[i,j] = summa[i,j] + (A[i,k]+A[k,j])
    temp2 = summa - (2 * A)
    F = temp2 * A
    return F

def np_edge_to_vertex(A):
    temp1 = np.sum(A + A.T, axis = 1)
    F = temp1 - (2 * np.diag(A))
    return F



for it in range(smallest_n, largest_n):
    n = it
    corpus = Corpus_n(n = n, target = 1, initial = 0)
    corpus.generate_random_graphs(N=N, total_random = True, verbose = False)
    print('-'*10 + ' Corpus done! ' + '-'*10)
    
    data_X = np.ones((N, n, n))
    data_labels = np.ones((N,2))
    for i in range(N): 
        data_X[i] = corpus.corpus_list[i].A # numpy array
        data_labels[i] = corpus.corpus_list[i].label # 2 dim np array, categorical
    

    data_X = data_X.reshape(N, n, n, 1) # [samples, rows, columns, channels]
    test_size = 0.1
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_labels, test_size=test_size)
    
    
    inputs = tf.keras.Input(shape=(n,n,1)) # shape?
    # 1. ETE
    x = tf.keras.layers.Lambda(edge_to_edge)(inputs)
    print(x)
    # 2. ETV
    x = tf.keras.layers.Lambda(edge_to_vertex)(x)
    print(x)
    x = tf.keras.layers.Conv2D(1, (3,3))(x)
    # 3. 3x3 Conv2D n times
    #for i in range(n):
    x = tf.keras.layers.Conv2D(1, (3,3))(x)
    # 4. Fully connected
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(3*n, activation='relu')(x) # according to Melnikov 2020
    x = tf.keras.layers.Dense(10, activation='relu')(x) # according to Melnikov 2020
    x = tf.keras.layers.Dense(2)(x)
    model = tf.keras.models.Model(inputs, x)
    
    
    # display
    if verbose:
        print('-'*10)
        model.summary()
        print('-'*10)
    
    # change loss func
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # validation split??
    history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=epochs, verbose=verbose)
    
    
    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
    
    print('\nTest accuracy:', test_acc)
    

    predictions = model.predict(X_test)

    if verbose:
        plt.ylim(0.0, 1.0)
        plt.plot(np.linspace(0.0, epochs, epochs), history.history['loss'],'--', color = ((it-smallest_n)/largest_n, 0.3, (it-smallest_n)/largest_n), label = 'loss for ' + str(it) + ' nodes')
        plt.plot(np.linspace(0.0, epochs, epochs), history.history['accuracy'],'-', color = ((it-smallest_n)/largest_n, 0.3, (it-smallest_n)/largest_n), label = 'accuracy for ' + str(it) + ' nodes')
        plt.xlabel('epochs')
        plt.ylabel('learning performance')
        plt.legend()