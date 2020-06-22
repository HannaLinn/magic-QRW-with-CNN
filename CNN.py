# -*- coding: utf-8 -*-
"""
Created on Tue Jun 7 10:27:43 2020

@author: hanna
"""

'''
Update: non Trainable ETV and ETE layer comparison with rest, also constraints
'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers, constraints
import numpy as np

import matplotlib.pyplot as plt
from corpus_random_percentage import *
from sklearn.model_selection import train_test_split
import time


def gen_data(n_max = 10, n = 5, N = 100, test_size = 0.1, val_test_size = 0.5, percentage = False):
    corpus = Corpus_n(n_max = n_max, target = 1, initial = 0)
    corpus.generate_random_graphs(N=N, n = n, verbose = False, percentage = percentage)
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

def kernel_init(shape, dtype=None):
    n = shape[0]
    z = np.zeros((n-1, n-1))
    o = np.ones((n-1,1))
    detector1 = np.concatenate((z, o, z), axis = 1)
    detector2 = np.ones((1, n*2-1))
    detector2[0,n-1] = 0
    detector = np.concatenate((detector1, detector2, detector1), axis = 0)
    weights = detector.reshape(shape)
    
    return tf.convert_to_tensor(weights, dtype=dtype)

class ETE_ETV_Net(tf.keras.Model):
    
    def __init__(self):
        super(ETE_ETV_Net, self).__init__()
    
    def build(self, ETE_ETV_layer = True, trainable_ETE_ETV = True, reg_lambdas = (0.0, 0.05), con_norm = 1., dropout_rate = 0.2):
        inputs = tf.keras.Input(shape=(n,n,1), batch_size = batch_size)

        if ETE_ETV_layer == 1:
            # ETE
            x = layers.ZeroPadding2D(padding=(n-1, n-1))(inputs)
            x = layers.Conv2D(1, (2*n-1,2*n-1), kernel_regularizer=regularizers.l1_l2(l1=reg_lambdas[0], l2=reg_lambdas[1]), trainable = trainable_ETE_ETV, kernel_constraint = constraints.max_norm(con_norm), kernel_initializer = kernel_init)(x)
            x = tf.math.multiply(x, inputs)
            
            # ETV
            y = layers.ZeroPadding2D(padding=(n-1, n-1))(inputs)
            y = layers.Conv2D(1, (2*n-1,2*n-1), kernel_regularizer=regularizers.l1_l2(l1=reg_lambdas[0], l2=reg_lambdas[1]), trainable = trainable_ETE_ETV, kernel_constraint = constraints.max_norm(con_norm), kernel_initializer = kernel_init)(y)
            y = tf.linalg.diag_part(y)
            y = tf.reshape(y, (batch_size, n, 1, 1))
            
            combined = tf.concat([x, y], axis = 2)

        # sequencial
        elif ETE_ETV_layer == 2:
            # ETE
            x = layers.ZeroPadding2D(padding=(n-1, n-1))(inputs)
            x = layers.Conv2D(1, (2*n-1,2*n-1), kernel_regularizer=regularizers.l1_l2(l1=reg_lambdas[0], l2=reg_lambdas[1]), trainable = trainable_ETE_ETV, kernel_constraint = constraints.max_norm(con_norm), kernel_initializer = kernel_init)(x)
            x = tf.math.multiply(x, inputs)
            
            # ETV
            y = layers.ZeroPadding2D(padding=(n-1, n-1))(x)
            y = layers.Conv2D(1, (2*n-1,2*n-1), kernel_regularizer=regularizers.l1_l2(l1=reg_lambdas[0], l2=reg_lambdas[1]), trainable = trainable_ETE_ETV, kernel_constraint = constraints.max_norm(con_norm), kernel_initializer = kernel_init)(y)
            y = tf.linalg.diag_part(y)
            y = tf.reshape(y, (batch_size, n, 1, 1))

        # 2 plain layers
        elif ETE_ETV_layer == 3:
            x = tf.keras.layers.Conv2D(n, (3,3))(inputs)
            combined = tf.keras.layers.Conv2D(n, (3,3))(x)

        # no initial convolution layers
        else: 
            combined = inputs
        
        z = tf.keras.layers.Conv2D(n, (3,3))(combined)
        
        z = tf.keras.layers.Dropout(dropout_rate)(z)
        
        # Dense part
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


test_size = 0.1
val_test_size = 0.5
dropout_rate = 0.2
y_upper = 10.0

if input('reload corpus? y/n ') == 'y':
    N = int(input('N: '))
    n_max = int(input('n_max: '))
    n = int(input('n (lower or eq. to n_max): '))
    percentage = float(input('percentage of q in corpus: '))
    gen_data(n_max = n_max, n = n, N = N, test_size = test_size, val_test_size = val_test_size, percentage = percentage)
    X_train, y_train, X_test, y_test, X_val, y_val = load_data('train_val_test_data.npz')
elif input('big run, N5000, n25, 50/50? y/n ') == 'y':
    X_train, y_train, X_test, y_test, X_val, y_val = load_data('train_val_test_dataN5000n_5050.npz')
    N = 5000
    n = 25
else:
    X_train, y_train, X_test, y_test, X_val, y_val = load_data('train_val_test_data.npz')
    n = X_train.shape[1]


if input('train? y/n ') =='y':
    batch_size = int(input('batch size (has to be a multiple of how many train and test samples): '))
    verbose = True
    epochs = int(input('epochs: '))
    colors = [(0.1, 0.1, 0.1), (0.9, 0.1, 0.4), (0.3, 0.2, 0.7), (0.2, 0.8, 0.6), (0.2, 0.6, 0.9)]

    '''
    Hyper parameters:
    model_list = [(ETE_ETV_layer, trainable_ETE_ETV, reg_lambdas = (l1, l2), con_norm)]
    1st: 0 no ete_etv_layer, 1 ete_etv_layer, 2 seq, 3 plain conv2d

    '''
    model_list = [(0),
                    (1, True, (0.15, 0.45), 1.),
                    (2, True, (0.15, 0.45), 1.),
                    (3, True),
                    (False)]

    for i in range(len(model_list)):
        model4 = ETE_ETV_Net()
        model4 = model4.build(model_list[i])
        start = time.time()
        history4 = model4.fit(X_train, y_train, batch_size=batch_size, validation_data = (X_val, y_val), epochs=epochs, verbose=verbose)
        end = time.time()
        vtime4 = end-start

        plt.figure(10)
        plt.title('All')
        plt.ylim(0.0, y_upper)
        plt.plot(np.linspace(0.0, epochs, epochs), history4.history['val_loss'],'--', color = colors[i], label = 'val loss for ' + str(n) + ' nodes')
        plt.plot(np.linspace(0.0, epochs, epochs), history4.history['val_accuracy'],'-', color = colors[i], label = 'val accuracy for ' + str(n) + ' nodes')
        plt.xlabel('epochs')
        plt.ylabel('learning performance')
        plt.legend()
        plt.savefig('All')

        plt.figure(i)
        plt.title(str(model_list[i]) + ' took time [min]: ' + str(vtime4/60))
        plt.ylim(0.0, y_upper)
        plt.plot(np.linspace(0.0, epochs, epochs), history4.history['val_loss'],'--', color = colors[i], label = 'val loss')
        plt.plot(np.linspace(0.0, epochs, epochs), history4.history['loss'],':', color = colors[i], label = 'loss')
        plt.plot(np.linspace(0.0, epochs, epochs), history4.history['val_accuracy'],'-', color = colors[i], label = 'val accuracy')
        plt.plot(np.linspace(0.0, epochs, epochs), history4.history['accuracy'],'-.', color = colors[i], label = 'accuracy')
        plt.xlabel('epochs')
        plt.ylabel('learning performance')
        plt.legend()
        plt.savefig('model ' + str(i))


    # ------------------------------------------------
    print('-'*20, ' DONE ', '-'*20)
    plt.show()