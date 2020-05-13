# -*- coding: utf-8 -*-
"""
Created on Mon May  4 09:57:23 2020
@author: hanna
"""

'''
Just an ANN, no convolutional filters.
Also wrong loss function.
Performs ok.
'''

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
from corpus_v3 import *
from sklearn.model_selection import train_test_split

verbose = True

for it in range(1):
    n = 5
    corpus = Corpus_n(n = n, target = 1, initial = 0)
    corpus.generate_graphs(line = True, cyclic = False)
    
    N = len(corpus.corpus_list)
    data_X = np.ones((N, n, n))
    data_labels = np.ones((N, 2))
    for i in range(N): 
        data_X[i] = corpus.corpus_list[i].A # numpy array
        data_labels[i] = corpus.corpus_list[i].quantum_advantage # Boolean
    
    # needs mixing, also use k-fold, cross validation?
    # split into train and test
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_labels, test_size=test_size)
    
    # architecture is wrong, needs filters/convolutional layers
    # Not do sequencial??
    model = keras.Sequential([
                            keras.layers.Flatten(input_shape=(n, n)),
                            keras.layers.Dense(128, activation='relu'),
                            keras.layers.Dense(2)])
    # change loss func
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    epochs = 20
    # validation split??
    history = model.fit(X_train, y_train, epochs=epochs, verbose=0)
    print(np.array(history.history['loss']).shape)
    
    
    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
    
    print('\nTest accuracy:', test_acc)
    
    probability_model = tf.keras.Sequential([model, 
                                             tf.keras.layers.Softmax()])
        
    predictions = probability_model.predict(X_test)
    
    got_right = 0
    i = 0
    for p in predictions:
        #print(np.round(p[0]))
        # print(y_test[i])
        if np.round(p[0]) != y_test[i]:
            got_right += 1
        i += 1
    print('\nGot right: ', got_right, ' out of ', len(y_test))
    
    if verbose:
        plt.ylim(0.0, 1.0)
        plt.plot(np.linspace(0.0, epochs, epochs), np.array(history.history['loss']), label = 'loss')
        plt.plot(np.linspace(0.0, epochs, epochs), np.array(history.history['accuracy']), label = 'accuracy')
        plt.xlabel('epochs')
        plt.ylabel('learning performance')
        