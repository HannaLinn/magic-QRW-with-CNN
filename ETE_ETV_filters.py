# -*- coding: utf-8 -*-
"""
Created on Fri May 15 07:44:43 2020

@author: hanna
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np

# Example
class Linear(layers.Layer):
    def __init__(self, units = 32, input_dim = 32):
        super(Linear, self).__init__()
        self.w = self.add_weight(shape = (input_dim, units),
                                       initializer = 'random_normal',
                                       trainable = True)
        self.b = self.add_weight(shape = (units,),
                                    initializer = 'zeros',
                                    trainable = True)
        
    def call(self, inputs): # change here
        return tf.matmul(inputs, self.w) + self.b
 
A = np.array([[0,0,0,1],
                 [0,0,1,1],
                 [0,1,0,0],
                 [1,1,0,0]], dtype='float32')


A_asym = np.array([[0,0,0,0],
                 [0,0,0,0],
                 [0,1,0,0],
                 [1,1,0,0]], dtype='float32')

class np_ETV(layers.Layer):
    def __init__(self, units = 32, input_dim = 32):
        super(np_ETV, self).__init__()
        self.w = self.add_weight(shape = (input_dim, units),
                                       initializer = 'identity',
                                       trainable = True)
        print(self.w)
        self.b = self.add_weight(shape = (units,),
                                    initializer = 'zeros',
                                    trainable = True)
        
    def call(self, inputs):
        A = inputs.numpy()
        n = len(A)
        temp1 = np.sum(A + A.T, axis = 1)
        F = temp1 - (2 * np.diag(A))
        F = F.reshape(1,n)
        F = tf.convert_to_tensor(F, dtype=tf.float32)
        return tf.matmul(F, self.w) + self.b


class np_ETE(layers.Layer):
    def __init__(self, units = 32, input_dim = 32):
        super(np_ETE, self).__init__()
        self.w = self.add_weight(shape = (input_dim, units),
                                       initializer = 'identity',
                                       trainable = True)
        self.b = self.add_weight(shape = (units,),
                                    initializer = 'zeros',
                                    trainable = True)
        
    def call(self, inputs):
        A = inputs.numpy()
        n = len(A)
        summa = np.zeros((n,n))
        for j in range(n):
            for i in range(n):
                for k in range(n):
                    summa[i,j] = summa[i,j] + (A[i,k]+A[k,j])
        temp2 = summa - (2 * A)
        F = temp2 * A
        F = tf.convert_to_tensor(F, dtype=tf.float32)
        return tf.matmul(F, self.w) + self.b

# ----------------------------------------------------------------------------


class tf_ETV(layers.Layer):
    def __init__(self, units = 32, input_dim = 32):
        super(ETV, self).__init__()
        self.w = self.add_weight(shape = (input_dim, units),
                                       initializer = 'identity',
                                       trainable = True)
        self.b = self.add_weight(shape = (units,),
                                    initializer = 'zeros',
                                    trainable = True)
        
    def call(self, inputs):
        A = inputs
        temp1 = tf.reduce_sum(A + tf.transpose(A), axis = 1)
        F = tf.reshape(temp1 - (2 * tf.linalg.tensor_diag_part(A)), (1, self.w.shape[1]))
        return tf.matmul(F, self.w) + self.b


class tf_ETE(layers.Layer):
    def __init__(self, units = 32, input_dim = 32):
        super(ETE, self).__init__()
        self.w = self.add_weight(shape = (input_dim, units),
                                       initializer = 'identity',
                                       trainable = True)
        self.b = self.add_weight(shape = (units,),
                                    initializer = 'zeros',
                                    trainable = True)
        
    def call(self, inputs):
        A = inputs
        n = self.w.shape[1]
        temp1 = tf.Variable(inputs)
        summa = np.zeros((n,n))
        for j in range(n):
            for i in range(n):
                for k in range(n):
                    summa[i,j] = summa[i,j] + (A[i,k]+A[k,j])
        temp1.assign(summa)
        temp2 = temp1 - (2 * A)
        F = temp2 * A
        return tf.matmul(F, self.w) + self.b 

''' 
testlayerETV = ETV(4, 4)
y = testlayerETV(A_asym)
print(y)

print('-'*20)

testlayerETE = ETE(4, 4)
y = testlayerETE(A)
print(y)

x = testlayerETE(A)
x = testlayerETV(x)
print(x)
'''

'''
Computes the sum of elements across dimensions of a tensor.
tf.math.reduce_sum(
    input_tensor, axis=None, keepdims=False, name=None
)

'''



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
    print(temp1)
    F = temp1 - (2 * np.diag(A))
    return F

'''
print()
print('ETE', np_edge_to_edge(A))
print()
print('ETV', np_edge_to_vertex(A_asym))
print()
print('ETE then ETV ', np_edge_to_vertex(np_edge_to_edge(A)))
'''
A2 = np.array([[0, 0, 1, 1, 1, 1],
               [0, 0, 1, 1, 1, 1],
               [1, 1, 0, 1, 1, 0],
               [1, 1, 1, 0, 1, 1],
               [1, 1, 1, 1, 0, 1],
               [1, 1, 0, 1, 1, 0]])
    
print('ETE', np_edge_to_edge(A2))
# --------------------------------------
'''
class figure_ETE(layers.Layer):
    def __init__(self, units = 32, input_dim = 32):
        super(figure_ETE, self).__init__()
        
        self.n = units
        z = np.zeros((self.n-1, self.n-1))
        o = np.ones((self.n-1,1))
        detector1 = np.concatenate((z, o, z), axis = 1)
        detector2 = np.ones((1, self.n*2-1))
        detector2[0,self.n-1] = 0
        detector = np.concatenate((detector1, detector2, detector1), axis = 0)
        weights = [detector, np.asarray([0.0])]
        self.w = tf.Variable(initial_value=weights, trainable=True)
        self.b = self.add_weight(shape = (units,),
                                    initializer = 'zeros',
                                    trainable = True)
        
    def call(self, inputs):
        x = layers.Conv2D(1, (2*self.n-1,2*self.n-1))(inputs)
        
        r = tf.math.multiply(x, inputs) + self.b
        return r 
'''
# ETE as figure 5
def model_ETE(n, N):
    data = A2
    n = len(data)
    N = 1
    data = data.reshape(N, n, n, 1)
    model = models.Sequential()    
    model.add(keras.Input(shape=(n, n, 1)))
    model.add(layers.ZeroPadding2D(padding=(n-1, n-1)))
    
    z = np.zeros((n-1, n-1))
    o = np.ones((n-1,1))
    detector1 = np.concatenate((z, o, z), axis = 1)
    detector2 = np.ones((1, n*2-1))
    detector2[0,n-1] = 0
    detector = np.concatenate((detector1, detector2, detector1), axis = 0)
    print(detector)
    detector = detector.reshape(2*n-1, 2*n-1, 1, 1)

    model.add(layers.Conv2D(1, (2*n-1,2*n-1)))
    weights = [detector, np.asarray([0.0])]
    model.set_weights(weights)

    
    
    yhat = model.predict(data) #should be outside
    
    yhat = tf.math.multiply(yhat, data) # I added this bc it should be there, needs to be moved before pred
    print(yhat)
    #return model

def model_ETV(n, N):
    data = A_asym
    n = len(data)
    N = 1
    data = data.reshape(N, n, n, 1) 
    model = models.Sequential()    
    model.add(keras.Input(shape=(n, n, 1)))
    model.add(layers.ZeroPadding2D(padding=(n-1, n-1)))

    z = np.zeros((n-1, n-1))
    o = np.ones((n-1,1))
    detector1 = np.concatenate((z, o, z), axis = 1)
    detector2 = np.ones((1, n*2-1))
    detector2[0,n-1] = 0
    detector = np.concatenate((detector1, detector2, detector1), axis = 0)
    
    detector = detector.reshape(2*n-1, 2*n-1, 1, 1)
    model.add(layers.Conv2D(1, (2*n-1,2*n-1)))
    weights = [detector, np.asarray([0.0])]
    model.set_weights(weights)
    
    
    yhat = model.predict(data) # should be outside
    
    yhat_return = tf.linalg.tensor_diag_part(yhat) # I added this bc it should be there, needs to be moved before pred
    print(yhat_return)
    #return model

model_ETE(4,1)
#model_ETV(4,1)

'''
TO DO:
    ETE as a layer with costum weights, there is an example somewhere
    het ETV to function by making the detector only follow the diagonal or only take out diagonal
    
'''
