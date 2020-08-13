# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 14:01:05 2020

@author: hanna
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers, constraints


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

    
    def build(self, n, batch_size, ETE_ETV_layer = True, trainable_ETE_ETV = True, reg_lambdas = (0.0, 0.05), con_norm = 1., dropout_rate = 0.2):
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
            

            out = tf.concat([x, y], axis = 2)

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