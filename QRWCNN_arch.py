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
        Returns a matrix that will delete the upper right triangle of another matrix when they are multiplied.
        '''
        n = shape[0]
        out = tf.linalg.band_part(tf.ones([n, n]), -1, 0)
        out = tf.reshape(out, shape)
        return out
    

class ETE_ETV_Net(tf.keras.Model):
    
    def __init__(self, n, num_classes = 2, net_type = 1, conv_learn = True, num_ETE = 2, num_neurons = 10, depth_of_dense=1, num_mse = 1, num_mee = 1):
        super(ETE_ETV_Net, self).__init__()
        self.n = n
        self.num_classes = num_classes
        self.num_channels = 4 #n
        self.net_type = net_type
        self.conv_learn = conv_learn # layer inbetween ETE and ETV
        self.num_ETE = num_ETE
        self.num_neurons = num_neurons
        self.depth_of_dense = depth_of_dense
        self.num_mse = num_mse
        self.num_mee = num_mee
        self.num_filters = num_ETE * self.num_channels + num_mse + num_mee


    def ETE(self, inputs, trainable_ETE_ETV):
        '''
        Edge-to-edge layer that uses the ETEV_filter from the Filter class.
        The convolutional layer counts how many neighboring edges each edge has.
        '''
        x = layers.Conv2D(1, (2*self.n-1, 2*self.n-1), padding='same', trainable = trainable_ETE_ETV, kernel_initializer = Filters.ETEV_kernel)(inputs)
        x = tf.math.multiply(x, inputs)
        return x

    def ETV(self, inputs, batch_size, trainable_ETE_ETV):
        '''
        Edge-to-vertex layer that uses the ETEV_filter from the Filter class.
        The convolutional layer summarizing information about the edges in the vertices.
        '''
        y = layers.Conv2D(1, (2*self.n-1, 2*self.n-1), padding='same', trainable = trainable_ETE_ETV, kernel_initializer = Filters.ETEV_kernel)(inputs)
        y = tf.linalg.diag_part(y[:,:,:,0]) # takes away channels
        y = tf.expand_dims(y, axis = -1) # adding channels
        return y

    def dense_layers(self, inputs):
        z = tf.keras.layers.Flatten()(inputs)
        z = tf.keras.layers.Dense(self.num_neurons, activation = 'relu')(z) # according to Melnikov code self.fc3, line 93 in cnn_arch
        for i in range(self.depth_of_dense):
            z = tf.keras.layers.Dense(self.num_neurons, activation = 'relu')(z)
        z = tf.keras.layers.Dense(self.num_classes, activation = 'softmax')(z)
        #z = tf.keras.layers.Dense(self.num_classes)(z)
        return z

    def build(self, batch_size = 1, reg_lambdas = (0.0, 0.0), con_norm = 1000., dropout_rate = 0.0):
        inputs = tf.keras.Input(shape=(self.n, self.n, 1))
        
        print('-'*20, 'net type: ', str(self.net_type), '-'*20)
        
        if self.net_type == 1:
            # 1st sequence
            for i_ete in range(self.num_ETE):
                x = inputs
                for j_ete in range(i_ete):
                    # ETE
                    x = self.ETE(x, trainable_ETE_ETV = False)
                # delete sym part
                x = tf.math.multiply(x, Filters.del_sym_part(shape=(self.n, self.n, 1)))
                
                
                # row 183 in mel
                if self.conv_learn:
                    x = layers.Conv2D(self.num_channels, 
                                        kernel_size = 3,
                                        padding = 'same',
                                        activation = 'relu',
                                        kernel_regularizer = regularizers.l1_l2(l1=reg_lambdas[0],l2=reg_lambdas[1]),
                                        kernel_constraint = constraints.max_norm(con_norm))(x)
                    
                    for i_channel in range(self.num_channels):
                        # ETV
                        x_temp = self.ETV(tf.reshape(x[:, :, :, i_channel], (batch_size, self.n, self.n, 1)), batch_size, trainable_ETE_ETV = False)
                        try:
                            x_out = layers.concatenate([x_out, x_temp], axis = 2)
                        except:
                            x_out = x_temp
                else:
                    # ETV
                    x = self.ETV(x, batch_size, trainable_ETE_ETV = False)
                    try:
                        x_out = layers.concatenate([x_out, x], axis = 2)
                    except:
                        x_out = x
            
            # 2nd sequence
            y = inputs
            # Mark start and mark start edge
            for i_mse in range(self.num_mse):
                y = tf.math.multiply(y, Filters.mark_start_filter(shape=(self.n, self.n, 1))) # MS
                for j_mse in range(i_mse):
                    y = tf.math.multiply(y, y) # MSE
                y = - y[:,:,0,:] # only want initial vertex 0
                try:
                    y_out = layers.concatenate([y_out, y], axis = 2)
                except:
                    y_out = y

            # 3rd sequence
            z = inputs
            # Mark end and Mark end edge
            for i_mee in range(self.num_mee):
                z = tf.math.multiply(z, Filters.mark_end_filter(shape=(self.n, self.n, 1))) # ME
                for j_mee in range(i_mee):
                    z = tf.math.multiply(z, z) # MEE
                z = - z[:,:,1,:] # only want target vertex 1
                try:
                    z_out = layers.concatenate([z_out, z], axis = 2)
                except:
                    z_out = z

            # put everything on a pile
            out = layers.concatenate([x_out, y_out, z_out], axis = 2)

            if dropout_rate != 0.0:
                out = tf.keras.layers.Dropout(dropout_rate)(out)
            
            # Dense part
            out = self.dense_layers(out)

        # 2 plain layers
        elif self.net_type == 2:
            print('plain Conv2D layers')
            x = inputs
            x = tf.keras.layers.Conv2D(self.n, (3,3))(inputs)
            x = tf.keras.layers.Conv2D(self.num_neurons, (3,3))(inputs)
            x = tf.keras.layers.Conv2D(self.num_neurons, (3,3))(inputs)
            out = tf.keras.layers.Conv2D(self.num_neurons, (3,3))(x)

            if dropout_rate != 0.0:
                out = tf.keras.layers.Dropout(dropout_rate)(out)
            
            # Dense part
            out = self.dense_layers(out)

        # no initial convolution layers
        else: 
            print('No conv layers!')
            z = tf.keras.layers.Flatten()(inputs)
            z = tf.keras.layers.Dense(self.num_neurons, activation = 'relu')(z) # according to Melnikov code self.fc3, line 93 in cnn_arch
            for i in range(self.depth_of_dense):
                z = tf.keras.layers.Dense(self.num_neurons, activation = 'relu')(z)
            out = tf.keras.layers.Dense(self.num_classes, activation = 'softmax')(z)

        model = tf.keras.models.Model(inputs, out)
        
        model.summary()
        
        #opt = keras.optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True)
        opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

        model.compile(optimizer = opt,
                      loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                      metrics = ['accuracy'])
        #model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])

        
        return model