
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 7 10:27:43 2020

@author: hanna
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from corpus_generator import *
from QRWCNN_arch import *
from tensorflow.keras.utils import plot_model
from util_functions import *


# Saving files
import os, inspect  # for current directory
current_file_directory = os.path.dirname(os.path.abspath(__file__))
from sklearn.metrics import f1_score, precision_score, recall_score

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#tf.config.list_physical_devices('GPU')

file_dir = current_file_directory + '/results_main'

n = 15
magic = True
names = ['c', 'q', 'positive', 'negative', 'T', 'H']
comp_list = [1, 3] 
num_classes = len(comp_list)
names = [names[x] for x in comp_list]

batch_size = 100
epochs = 2000
average_num = 10

'''
*Init* : n,
num_classes = 2

net_type = 1,
conv_learn = False,
num_ETE = 2,
num_neurons = 10,
depth = 1,
num_mse = 1,
num_mee = 1.

*Build* : batch_size = 1,
reg_lambdas = (0.0, 0.0),
con_norm = 1000.,
dropout_rate = 0.0.

3 first change num_ETE
4th non trainable ETE_layer
5 - 6th change num_neurons
7th try regulisation lasso and ridge
8th try regulisation condition norm of the weights
9th try regularisation by dropout
10th plain dense network
'''

model_list = [[(1, True, 2, 10, 3), ((0.0, 0.0), 1000., 0.2)]]
now_testing = '_plainCONV_network_lr_0_001_largerdataset_epochs2000_bz100_n10'


if not magic:
    data_X, data_labels = load_data( current_file_directory + '/results_nomagic_corpus' + '/train_val_test_data_n_' + str(n) + '.npz')    
else:
    data_X, data_labels, data_ranking = load_data( current_file_directory + '/results_magic_corpus_ranking' + '/train_val_test_data_n_' + str(n) + '.npz')    
    data_labels = choose_types(comp_list, data_ranking)

#data_X, data_labels, data_ranking = load_data(file_dir + '/train_val_test_data_big.npz')
print('Before rinse (N, n, n, C), labels: ', data_X.shape, data_labels.shape,)

num_classes = len(comp_list)
if magic:
    file_dir = current_file_directory + '/results_main_magic'


data_X, data_labels = delete_ties(data_X, data_labels)
data_X, data_labels = rinse_out(data_X, data_labels, np.array([1., 0.]), 0)

print('\nData spec.:')
print('N_train: ', data_X.shape[0])
q_count_train = (data_labels[:,1] == 1.0).sum()
print(names[1], '_percentage: %.2f' % (q_count_train/data_labels.shape[0]))
input()

grande_train_loss = np.zeros((epochs, average_num))
grande_train_accuracy = np.zeros((epochs, average_num))
grande_test_accuracy = np.zeros((epochs, average_num))
grande_test_loss = np.zeros((epochs, average_num))
grande_precision = np.zeros((num_classes, average_num))
grande_recall = np.zeros((num_classes, average_num))
grande_f1 = np.zeros((num_classes, average_num))
i = 0
    
for average_iter in range(average_num):
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_labels, test_size=0.3)

    X_train, y_train = div_by_batch_size(X_train, y_train, batch_size)
    X_test, y_test = div_by_batch_size(X_test, y_test, batch_size)

    print('(N, n, n, C): ', X_train.shape)

    print('\ntrain spec.:')
    print('N_train: ', X_train.shape[0])
    q_count_train = (y_train[:,1] == 1.0).sum()
    print(names[1], '_percentage: %.2f' % (q_count_train/X_train.shape[0]))


    print('\nTest spec.:')
    print('N_test: ', X_test.shape[0])
    q_count_test = (y_test[:,1] == 1.0).sum()
    print(names[1],'_percentage: %.2f' %(q_count_test/X_test.shape[0]))

    assert not np.any(np.isnan(y_test))
    assert np.sum(y_test) == y_test.shape[0] # any ties

    y_upper = 1.0

    assert len(X_train) % batch_size == 0
    assert len(X_test) % batch_size == 0
    assert np.all(np.sum(y_train, axis = 1) == 1.) # no ties


    validation_freq = 1

    file2 = open(file_dir +'/train_param.txt', 'w')
    q_count_train = (y_train[:,1] == 1.0).sum()
    q_count_test = (y_test[:,1] == 1.0).sum()
    L = ['batch_size: ' + str(batch_size) + '\n', 'epochs: ' + str(epochs) + '\n', 'train (N, n, n, C): ' + str(X_train.shape)+ ' q_percentage: %.2f' % (q_count_train/X_train.shape[0]) + '\n', 'test (N, n, n, C): ' + str(X_test.shape)+ ' q_percentage: %.2f' %(q_count_test/X_test.shape[0]) + '\n']
    file2.writelines(L)
    file2.close()

    X_train, y_train, X_test, y_test, = tf.convert_to_tensor(X_train), tf.convert_to_tensor(y_train), tf.convert_to_tensor(X_test), tf.convert_to_tensor(y_test)

    print('-'*20, ' average iter: ', str(average_iter), ' n: ', str(n), '-'*20)

    model = ETE_ETV_Net(n, num_classes, *model_list[i][0])
    model = model.build(batch_size, *model_list[i][1])
    #plot_model(model, to_file=file_dir + 'model_plot' + str(i) + 'm.png', show_shapes=True, show_layer_names=True)
    start = time.time()
    callbacks = [tf.keras.callbacks.TerminateOnNaN()]
    history = model.fit(X_train, y_train, callbacks=callbacks, batch_size=batch_size, validation_data = (X_test, y_test), validation_freq = validation_freq, epochs=epochs, verbose=2, shuffle = True)        
    
    end = time.time()
    vtime4 = end-start

    y_pred1 = model.predict(X_test, batch_size=batch_size)
    y_pred=np.eye(1, num_classes, k=np.argmax(y_pred1, axis =1)[0])
    for pred in range(1, y_pred1.shape[0]):
        y_pred = np.append(y_pred, np.eye(1, num_classes, k=np.argmax(y_pred1, axis = 1)[pred]), axis = 0)

    grande_train_loss[:, average_iter] = history.history['loss']
    grande_test_accuracy[:, average_iter] = history.history['val_accuracy']
    grande_train_accuracy[:, average_iter] = history.history['accuracy']
    grande_test_loss[:, average_iter] = history.history['val_loss']
    grande_precision[:, average_iter] = precision_score(y_test, y_pred, average=None)
    grande_recall[:, average_iter] = recall_score(y_test, y_pred, average=None)
    grande_f1[:, average_iter] = f1_score(y_test, y_pred, average=None)

    plt.figure(1)
    plt.title(str(model_list) + ' took time [min]: ' + str(round(vtime4/60,3)) + str(names) + 'n: ' + str(n))
    plt.ylim(0.0, y_upper)
    y = history.history['val_loss']
    plt.plot(np.linspace(0.0, len(y), len(y)), y,'--', label = 'test loss')
    y = history.history['loss']
    plt.plot(np.linspace(0.0, len(y), len(y)), y,':', label = 'train loss')
    y = history.history['val_accuracy']
    plt.plot(np.linspace(0.0, len(y), len(y)), y,'-', label = 'test accuracy')
    y = history.history['accuracy']
    plt.plot(np.linspace(0.0, len(y), len(y)), y,'-.', label = 'train accuracy')
    plt.xlabel('epochs')
    plt.ylabel('learning performance')
    plt.legend()
    plt.savefig(file_dir +'/model' + now_testing)
    np.savez(file_dir +'/train_results', history.history['loss'], history.history['val_accuracy'], history.history['val_loss'])



train_loss = np.average(grande_train_loss, 1)
test_accuracy = np.average(grande_test_accuracy, 1)
train_accuracy = np.average(grande_train_accuracy, 1)
test_loss = np.average(grande_test_loss, 1)
precision = np.average(grande_precision, 1)
recall = np.average(grande_recall, 1)
f1 = np.average(grande_f1, 1)

file1 = open(file_dir +'/f1_precision_recall.txt', 'w')
L = ['model ' + str(i) + '\n' + 
'precision: ' + str(precision_score(y_test, y_pred, average=None))+ '\n' +
'recall: ' + str(recall_score(y_test, y_pred, average=None)) +'\n' + 
'f1: ' + str(f1_score(y_test, y_pred, average=None)) + '\n\n\n']
file1.writelines(L)
file1.close()

plt.figure(2)
plt.title(str(model_list[i]) + ' random with types '  + str(names) + 'n: ' + str(n))
plt.ylim(0.0, y_upper)
plt.plot(np.linspace(0.0, len(train_loss), len(train_loss)), train_loss,':', label = 'train loss ' + str(round(train_loss[-1],2)))
plt.plot(np.linspace(0.0, len(test_accuracy), len(test_accuracy)), test_accuracy,'-', label = 'test accuracy ' + str(round(test_accuracy[-1],2)))
plt.plot(np.linspace(0.0, len(train_accuracy), len(train_accuracy)), train_accuracy,'-.', label = 'train accuracy ' + str(round(train_accuracy[-1],2)))
plt.plot(np.linspace(0.0, len(test_loss), len(test_loss)), test_loss,'--', label = 'test loss ' + str(round(test_loss[-1],2)))
plt.xlabel('epochs')
plt.ylabel('learning performance')
plt.legend()
np.savez(file_dir +'/train_results' + str(n), train_loss, test_accuracy, train_loss, test_loss)

plt.savefig(file_dir+'/random_graphs_' + str(average_num) + now_testing)

# ------------------------------------------------
print('-'*20, ' DONE ', '-'*20)
plt.show()
