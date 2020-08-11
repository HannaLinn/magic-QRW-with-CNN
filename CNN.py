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
from corpus_generator import *
from sklearn.model_selection import train_test_split
import time


def gen_data(n_max = 10, n = 5, N = 100, test_size = 0.1, val_test_size = 0.5, percentage = False, random = True, line = False, cyclic = False):
    corpus = Corpus_n(n_max = n_max, target = 1, initial = 0)
    corpus.generate_graphs(n = n, N = N, verbose = 0, percentage = percentage, random = random, line = line, cyclic = cyclic)
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


class Filters():
    def ETEV_kernel(shape, dtype=None):
        n = shape[0]
        z = np.zeros((n-1, n-1))
        o = np.ones((n-1,1))
        detector1 = np.concatenate((z, o, z), axis = 1)
        detector2 = np.ones((1, n*2-1))
        detector2[0,n-1] = 0
        detector = np.concatenate((detector1, detector2, detector1), axis = 0)
        weights = detector.reshape(shape)
        return tf.convert_to_tensor(weights, dtype=dtype)

    def mark_start_filter(shape, dtype=None):
        mark = 0
        n = shape[0]
        out = np.ones([n, n])
        out[mark, :] = -1
        out[:, mark] = -1
        out = tf.convert_to_tensor(out)
        out = tf.reshape(out, (1,1,n,n))
        return out

    def mark_end_filter(shape, dtype=None):
        mark = 1
        n = shape[0]
        out = np.ones([n, n])
        out[mark, :] = -1
        out[:, mark] = -1
        out = tf.convert_to_tensor(out)
        out = tf.reshape(out, (1,1,n,n))
        return out

    def del_sym_part(shape, dtype=None):
        n = shape[0]
        out = tf.linalg.band_part(tf.ones([n, n]), -1, 0)
        out = tf.reshape(out, (1,1,n,n))
        return out

    def identity_filter(shape, dtype=None):
        n = shape[0]
        out = np.zeros((2*n-1, 2*n-1))
        out[n-1, n-1] = 1
        out = tf.convert_to_tensor(out)
        out = tf.reshape(out, (1, 1, 2*n-1, 2*n-1))
        return out



class ETE_ETV_Net(tf.keras.Model):
    
    def __init__(self, n):
        super(ETE_ETV_Net, self).__init__()
        self.n = n

    def ETE(self, inputs, trainable_ETE_ETV, reg_lambdas, con_norm, dropout_rate):
        x = layers.ZeroPadding2D(padding=(self.n-1, self.n-1))(inputs)
        x = layers.Conv2D(1, (2*self.n-1,2*self.n-1), kernel_regularizer=regularizers.l1_l2(l1=reg_lambdas[0], l2=reg_lambdas[1]), trainable = trainable_ETE_ETV, kernel_constraint = constraints.max_norm(con_norm), kernel_initializer = Filters.ETEV_kernel)(x)
        x = tf.math.multiply(x, inputs)
        return x

    def ETV(self, inputs, trainable_ETE_ETV, reg_lambdas, con_norm, dropout_rate):
        y = layers.ZeroPadding2D(padding=(self.n-1, self.n-1))(inputs)
        y = layers.Conv2D(1, (2*self.n-1,2*self.n-1), kernel_regularizer=regularizers.l1_l2(l1=reg_lambdas[0], l2=reg_lambdas[1]), trainable = trainable_ETE_ETV, kernel_constraint = constraints.max_norm(con_norm), kernel_initializer = Filters.ETEV_kernel)(y)
        y = tf.linalg.diag_part(y)
        y = tf.reshape(y, (batch_size, self.n, 1, 1))
        return y

    def dense_layers(self, inputs):
        z = tf.keras.layers.Flatten()(inputs)
        z = tf.keras.layers.Dense(3*self.n, activation='relu')(z) # according to Melnikov 2020
        z = tf.keras.layers.Dense(10, activation='relu')(z) # according to Melnikov 2020
        z = tf.keras.layers.Dense(2)(z)
        return z

    
    def build(self, ETE_ETV_layer = True, trainable_ETE_ETV = True, reg_lambdas = (0.0, 0.05), con_norm = 1., dropout_rate = 0.2):
        inputs = tf.keras.Input(shape=(n,n,1), batch_size = batch_size)

        if ETE_ETV_layer == 1:
            # ETE
            # more than one time??
            x = self.ETE(inputs, self.n, trainable_ETE_ETV, reg_lambdas, con_norm, dropout_rate)
            
            # delete sym part
            x = tf.math.multiply(x, Filters.del_sym_part(shape=(n,n,1)))

            # one additional vanilla conv2d here?

            # ETV
            y = self.ETV(inputs, self.n, trainable_ETE_ETV, reg_lambdas, con_norm, dropout_rate)
            

            combined = tf.concat([x, y], axis = 2)

        # sequencial
        elif ETE_ETV_layer == 2:
            # ETE
            
            x = self.ETE(inputs, self.n, trainable_ETE_ETV, reg_lambdas, con_norm, dropout_rate)
            # more than one time?? 3 times in Mel cnn_arch line 116
            num_ETE = 3
            for i in range(num_ETE-1):
                x = self.ETE(x, self.n, trainable_ETE_ETV, reg_lambdas, con_norm, dropout_rate)
                
            # delete sym part
            x = tf.math.multiply(x, Filters.del_sym_part(shape=(n,n,1)))

            # one additional vanilla conv2d here?

            # ETV
            x = self.ETV(x, self.n, trainable_ETE_ETV, reg_lambdas, con_norm, dropout_rate)

            # Mark start and mark start edge 
            x = tf.math.multiply(x, Filters.mark_start_filter(shape=(self.n, self.n,1)))

            # Mark end and Mark end edge
            out = tf.math.multiply(x, Filters.mark_end_filter(shape=(self.n, self.n,1)))
            

        # 2 plain layers
        elif ETE_ETV_layer == 3:
            x = tf.keras.layers.Conv2D(n, (3,3))(inputs)
            out = tf.keras.layers.Conv2D(n, (3,3))(x)

        # no initial convolution layers
        else: 
            out = inputs
        
        z = tf.keras.layers.Conv2D(n, (3,3))(out)
        
        z = tf.keras.layers.Dropout(dropout_rate)(z)
        
        # Dense part
        z = self.dense_layers(z)

        model = tf.keras.models.Model(inputs, z)
        
        model.summary()
        

        #model.compile(optimizer='adam',
        #              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        #              metrics=['accuracy'])
        sgd = keras.optimizers.SGD(lr=0.001, decay=0., momentum=0.9)
        model.compile(optimizer='sgd',
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        
        return model


test_size = 0.1
val_test_size = 0.5
dropout_rate = 0.2
y_upper = 2.0

if input('reload corpus? y/n ') == 'y':
    N = int(input('N: '))
    n_max = int(input('n_max: '))
    n = int(input('n (lower or eq. to n_max): '))
    percentage = float(input('percentage of q in corpus: '))
    random = int(input('random? 1/0 '))
    line = int(input('line? 1/0 '))
    gen_data(n_max = n_max, n = n, N = N, test_size = test_size, val_test_size = val_test_size, percentage = percentage, random = False, line = True)
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

    
    model_list = [(0),
                    (1, True, (0.15, 0.45), 1.),
                    (2, True, (0.15, 0.45), 1.),
                    (3, True),
                    (False)]
    '''
    model_list = [(2, True, (0.15, 0.45), 1.),
                (3, True),
                (False)]

    for i in range(1):
        model4 = ETE_ETV_Net(n)
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