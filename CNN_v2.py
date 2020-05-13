# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:39:29 2020

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

largest_n = 8 # n is number if nodes in the graphs
smallest_n = 4
epochs = 20
N=10 # number of random generated graphs

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
    
    test_size = 0.1
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_labels, test_size=test_size)
    
    # architecture is wrong, needs filters/convolutional layers
    model = models.Sequential()
    #model.add(layers.Lambda(lambda x: x**2, input_shape=(n,n)))
    model.add(layers.Flatten(input_shape=(n,n)))
    model.add(layers.Dense(3*n, activation='relu')) # according to Melnikov 2020
    model.add(layers.Dense(10, activation='relu')) # according to Melnikov 2020
    model.add(layers.Dense(2))
    
    # display
    if verbose:
        print('-'*10)
        model.summary()
        print('-'*10)
    
    # change loss func
    model.compile(optimizer='adam',
                  #loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # validation split??
    history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=epochs, verbose=verbose)
    
    
    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
    
    print('\nTest accuracy:', test_acc)
    
    # softmax to get predictions in probabilities
    probability_model = tf.keras.Sequential([model, 
                                             tf.keras.layers.Softmax()])
        
    predictions = probability_model.predict(X_test)
    
    got_right = 0
    i = 0
    for p in predictions:
        if np.round(p[0]) == y_test[i][0]:
            got_right += 1
        i += 1
    print('\nGot right: ', got_right, ' out of ', len(y_test))
    
    if verbose:
        plt.ylim(0.0, 1.0)
        plt.plot(np.linspace(0.0, epochs, epochs), history.history['loss'],'--', color = ((it-smallest_n)/largest_n, 0.3, (it-smallest_n)/largest_n), label = 'loss for ' + str(it) + ' nodes')
        plt.plot(np.linspace(0.0, epochs, epochs), history.history['accuracy'],'-', color = ((it-smallest_n)/largest_n, 0.3, (it-smallest_n)/largest_n), label = 'accuracy for ' + str(it) + ' nodes')
        plt.xlabel('epochs')
        plt.ylabel('learning performance')
        plt.legend()