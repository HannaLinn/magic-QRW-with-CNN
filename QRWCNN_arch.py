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
        '''
        Edge-to-Edge as well as Edge-to-Vertex filter kernel.
        '''
        n = int((shape[0]+1)/2)
        z = np.zeros((n-1, n-1))
        o = np.ones((n-1,1))
        detector1 = np.concatenate((z, o, z), axis = 1)
        detector2 = np.ones((1, n*2-1))
        detector2[0, n-1] = 0
        detector = np.concatenate((detector1, detector2, detector1), axis = 0)
        weights = detector.reshape(shape)
        return tf.convert_to_tensor(weights, dtype=dtype)

    def mark_start_filter(shape, dtype=None):
        '''
        Marking start, initial node for the random walk. Hard coded as initial = 0.
        Mark the nearest edges to the initial vertex.
        '''
        mark = 0
        n = shape[0]
        out = np.ones([n, n])
        out[mark, :] = -1
        out[:, mark] = -1
        out = tf.convert_to_tensor(out, dtype=tf.float32)
        out = tf.reshape(out, shape)
        return out

    def mark_end_filter(shape, dtype=None):
        '''
        Marking end, target node for the random walk. Hard coded as target = 1.
        Mark the nearest edges to the target vertex.
        '''
        mark = 1
        n = shape[0]
        out = np.ones([n, n])
        out[mark, :] = -1
        out[:, mark] = -1
        out = tf.convert_to_tensor(out, dtype=tf.float32)
        out = tf.reshape(out, shape)
        return out

    def del_sym_part(shape, dtype=None):
        '''
        ----N
        Returns a matrix that will delete the left lower triangle of another matrix when they are multiplied.
        '''
        n = shape[0]
        out = tf.linalg.band_part(tf.ones([n, n]), -1, 0)
        out = tf.reshape(out, shape)
        return out
    

class ETE_ETV_Net(tf.keras.Model):
    
    def __init__(self, n, num_classes):
        super(ETE_ETV_Net, self).__init__()
        self.n = n
        self.num_classes = num_classes
        self.num_kernels = n

    def ETE(self, inputs, trainable_ETE_ETV):
        '''
        Edge-to-edge layer that uses the ETEV_filter from the Filter class.
        The convolutional layer counts how many neighboring edges each edge has.
        '''
        x = layers.ZeroPadding2D(padding=(self.n-1, self.n-1))(inputs)
        x = layers.Conv2D(1, (2*self.n-1, 2*self.n-1), trainable = trainable_ETE_ETV, kernel_initializer = Filters.ETEV_kernel)(x)
        x = tf.math.multiply(x, inputs)
        return x

    def ETV(self, inputs, batch_size, trainable_ETE_ETV):
        '''
        Edge-to-vertex layer that uses the ETEV_filter from the Filter class.
        The convolutional layer summarizing information about the edges in the vertices.
        '''
        y = layers.ZeroPadding2D(padding=(self.n-1, self.n-1))(inputs)
        y = layers.Conv2D(1, (2*self.n-1,2*self.n-1), trainable = trainable_ETE_ETV, kernel_initializer = Filters.ETEV_kernel)(y)
        y = tf.linalg.diag_part(y)
        y = tf.reshape(y, (batch_size, self.n, 1, 1))
        return y

    def dense_layers(self, inputs, num_neurons, num_ETE):
        z = tf.keras.layers.Flatten()(inputs)
        z = tf.keras.layers.Dense(3*self.n, activation='relu')(z) # according to Melnikov 2020, more logical one if we wanna look at the weights
        #z = tf.keras.layers.Dense(num_ETE*4+2, activation='relu')(z) # according to Melnikov code self.fc1, line 89 cnn_arch     
        z = tf.keras.layers.Dense(num_neurons, activation='relu')(z) # according to Melnikov 2020
        z = tf.keras.layers.Dense(self.num_classes)(z)
        return z

    def build(self, batch_size, net_type = 1, conv = False, trainable_ETE_ETV = False, num_ETE = 2, num_neurons = 10, reg_lambdas = (0.0, 0.0), con_norm = 1000., dropout_rate = 0.0):
        inputs = tf.keras.Input(shape=(self.n, self.n, 1), batch_size=batch_size)
        
        print('-'*20, 'net type: ', str(net_type), '-'*20)
        
        if net_type == 1:
            # 1st sequence
            x = inputs
            for i_ete in range(num_ETE):
                for e_ete_iter in range(i_ete):
                    # ETE
                    x = self.ETE(x, trainable_ETE_ETV)
                # delete sym part
                x = tf.math.multiply(x, Filters.del_sym_part(shape=(self.n, self.n, 1)))
                
                
                # row 183 in mel
                if conv:
                    x = layers.Conv2d(self.num_kernels, 
                                        kernel_size=3,
                                        padding='valid',
                                        activation='relu',
                                        kernel_regularizer=regularizers.l1_l2(l1=reg_lambdas[0],l2=reg_lambdas[1]),
                                        kernel_constraint = constraints.max_norm(con_norm))
                
                    for i_kernel in range(self.num_kernels):
                        # ETV
                        x = self.ETV(x, batch_size, trainable_ETE_ETV)
                else:
                    # ETV
                    x = self.ETV(x, batch_size, trainable_ETE_ETV)
            
            # 2nd sequence
            y = inputs
            # Mark start and mark start edge 
            y = tf.math.multiply(y, Filters.mark_start_filter(shape=(self.n, self.n, 1))) # MS
            y = tf.math.multiply(y, y) # MSE

            # 3rd sequence
            z = inputs
            # Mark end and Mark end edge
            out = tf.math.multiply(z, Filters.mark_end_filter(shape=(self.n, self.n, 1))) # ME
            z = tf.math.multiply(z, z) # MEE

            # put everything on a pile
            out = layers.concatenate([x, y, z], axis = 2)

        # 2 plain layers
        elif net_type == 2:
            print('plain layers')
            x = tf.keras.layers.Conv2D(self.n, (3,3))(inputs)
            out = tf.keras.layers.Conv2D(self.n, (3,3))(x)

        # no initial convolution layers
        else: 
            print('No conv layers!')
            input()
            out = inputs
        

        if dropout_rate != 0.0:
            out = tf.keras.layers.Dropout(dropout_rate)(out)
        
        # Dense part
        out = self.dense_layers(out, num_neurons, num_ETE)

        model = tf.keras.models.Model(inputs, out)
        
        model.summary()
        
        sgd = keras.optimizers.SGD(lr=0.001, decay=0., momentum=0.9)

        model.compile(optimizer='sgd',
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        #model.run_eagerly=True
        #model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])

        
        return model