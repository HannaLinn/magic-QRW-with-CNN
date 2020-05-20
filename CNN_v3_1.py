# -*- coding: utf-8 -*-
"""
Created on Mon May 18 08:57:50 2020

@author: hanna
"""

'''
Eq 6 and 7 in preprocessing.
n conv filters fixed.

TO DO:
3. single batch of 3 examples per epoch, translated to batch size of 3
4. val test set
'''

import tensorflow as tf
from tensorflow.keras import layers, models

import numpy as np
import matplotlib.pyplot as plt
from corpus_v4 import *
from sklearn.model_selection import train_test_split

verbose = True

largest_n = 8 # n is number if nodes in the graphs
smallest_n = 5
epochs = 200
N=1000 # number of random generated graphs
#filters = 100

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
        x = corpus.corpus_list[i].A
        #print(x)
        # 1. ETE
        #for j in range(filters): does not work, values gets too large
        x = np_edge_to_edge(x) # does not help, same accuracy
        #print(x)
        # 2. ETV
        #for j in range(filters): does not work, values gets too large
        x = np_edge_to_vertex(x) # does not help, same accuracy
        #print(x)
        data_X[i] = x # numpy array
        data_labels[i] = corpus.corpus_list[i].label # 2 dim np array, categorical
    

    data_X = data_X.reshape(N, n, n, 1) # [samples, rows, columns, channels]
    test_size = 0.1
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_labels, test_size=test_size)
    val_test_size = 0.5
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=val_test_size)
    
    inputs = tf.keras.Input(shape=(n,n,1))

    # 3. 3x3 Conv2D n times
    x = tf.keras.layers.Conv2D(n, (3,3))(inputs)
    
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
    history = model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs=epochs, verbose=verbose)
    
    
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