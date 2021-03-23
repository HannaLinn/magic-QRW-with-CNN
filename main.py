# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 14:00:59 2021

@author: hanna
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from corpus_generator import *
from QRWCNN_arch import *
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from sklearn.metrics import f1_score, precision_score, recall_score
from util_functions import *

# Saving files
import os, inspect  # for current directory
current_file_directory = os.path.dirname(os.path.abspath(__file__))

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#tf.config.list_physical_devices('GPU')

names = ['c', 'q', 'positive', 'negative', 'T', 'H']
comp_list = [0, 1] 
num_classes = len(comp_list)
names = [names[x] for x in comp_list]

magic = False
file_name = current_file_directory + '/results'


y_upper = 1.0 # for plotting
epochs = 10
colors = [(51, 51, 51), (76, 32, 110), (32, 92, 25), (32, 95, 105), (110, 32, 106), (189, 129, 9)]# ['grey', 'purple', 'green', 'blue', 'magenta', 'orange']
colors = [tuple(t/250 for t in x) for x in colors] # only relevant colors

validation_freq = 1
average_num = 1
n_min_loop = 5
n_max_loop = 6
training_time = np.zeros((n_max_loop-n_min_loop,average_num))
reg = (0.0, 0.0)
dropout_rate = 0.0
batch_size = 10
net_type = 1
N = 10000


'''
*Init ANN* :
n,
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
'''

model_list = [[(1, True, 2, 10, 3), ((0.0, 0.0), 1000., 0.2)]]


file1 = open(file_name +'/f1_precision_recall.txt', 'w')
for n_iter in range(n_min_loop, n_max_loop):

    grande_training_loss = np.zeros((epochs, average_num))
    grande_test_accuracy = np.zeros((epochs, average_num))
    grande_test_loss = np.zeros((epochs, average_num))
    grande_precision = np.zeros((num_classes, average_num))
    grande_recall = np.zeros((num_classes, average_num))
    grande_f1 = np.zeros((num_classes, average_num))
    
    for average_iter in range(average_num):
        print('-'*20, ' average iter: ', str(average_iter), ' n: ', str(n_iter), '-'*20)
        try:
            if magic:
                data_X, data_labels, data_ranking = load_data(current_file_directory + '/linear_corpuses_magic' +'/train_val_test_data'+str(n_iter)+'.npz')
            else:
                data_X, data_labels, data_ranking = load_data(current_file_directory + '/linear_corpuses' +'/train_val_test_data'+str(n_iter)+'.npz')
        except:
            if magic:
                gen_data(N = N, n_max = n_iter, n = n_iter, linear = True, cyclic = False, all_graphs = False, duplicates = False, magic = True)
                data_X, data_labels, data_ranking = load_data(current_file_directory + '/linear_corpuses_magic'+'/train_val_test_data'+str(n_iter)+'.npz')
            else:
                gen_data(n_max = n_iter, n = n_iter, linear = True, cyclic = False, all_graphs = False, duplicates = True, magic = magic)
                data_X, data_labels, data_ranking = load_data(current_file_directory + '/linear_corpuses'+'/train_val_test_data'+str(n_iter)+'.npz')

        print(data_X.shape[0])
        data_labels = choose_types(comp_list, data_ranking)
        data_X, data_labels = delete_ties(data_X, data_labels)
        #data_X, data_labels = rinse_out(data_X, data_labels, np.array([1., 0.]), 4000)      
        print(data_X.shape[0])

        test_size = 0.1
        X_train, X_test, y_train, y_test = train_test_split(data_X, data_labels, test_size=test_size)
        print('N_train: ', X_train.shape[0])
        print('N_test: ', X_test.shape[0])
        X_train, y_train = div_by_batch_size(X_train, y_train, batch_size)
        X_test, y_test = div_by_batch_size(X_test, y_test, batch_size)

        print('\ntrain spec.:')
        print('N_train: ', X_train.shape[0])
        q_count_train = (y_train[:,1] == 1.0).sum()
        print(names[1], '_percentage: %.2f' % (q_count_train/X_train.shape[0]))
        #input()

        file2 = open(file_name +'/train_param' + str(n_iter) +'.txt', 'w')
        q_count_train = (y_train[:,1] == 1.0).sum()
        q_count_test = (y_test[:,1] == 1.0).sum()
        L = ['batch_size: ' + str(batch_size) + '\n', 'epochs: ' + str(epochs) + '\n', 'train (N, n, n, C): ' + str(X_train.shape)+ ' q_percentage: %.2f' % (q_count_train/X_train.shape[0]) + '\n', 'test (N, n, n, C): ' + str(X_test.shape)+ ' q_percentage: %.2f' %(q_count_test/X_test.shape[0]) + '\n']
        file2.writelines(L)
        file2.close()

        X_train, y_train, X_test, y_test = tf.convert_to_tensor(X_train), tf.convert_to_tensor(y_train), tf.convert_to_tensor(X_test), tf.convert_to_tensor(y_test)

        print('N ', X_train.shape)
        print('N ', X_test.shape)

        assert not np.any(np.isnan(y_test))
        assert np.all(np.sum(y_train, axis = 1) == 1.) # no ties
        num_classes = y_test.shape[1]

        model = ETE_ETV_Net(n_iter, num_classes, *model_list[0][0])
        model = model.build(batch_size, *model_list[0][1])
        #plot_model(model, to_file=file_name + '/model_plot'+ str(n_iter) +'n.png', show_shapes=True, show_layer_names=True)
        start = time.time()
        history = model.fit(X_train, y_train, batch_size=batch_size, steps_per_epoch = 3, validation_data = (X_test, y_test), validation_freq = validation_freq, epochs=epochs, verbose=2, shuffle=True)

        #history = model.fit(X_train, y_train, batch_size=batch_size, validation_data = (X_test, y_test), validation_freq = validation_freq, epochs=epochs, verbose=2, shuffle = True)

        end = time.time()
        vtime = end-start
        training_time[n_iter-n_min_loop, average_iter] = vtime


        y_pred1 = model.predict(X_test, batch_size=batch_size)
        y_pred=np.eye(1, num_classes, k=np.argmax(y_pred1, axis = 1)[0])
        for i in range(1, y_pred1.shape[0]):
            y_pred = np.append(y_pred, np.eye(1, num_classes, k=np.argmax(y_pred1, axis = 1)[i]), axis = 0)

        training_loss = history.history['loss']
        test_loss = history.history['val_loss']
        test_accuracy = history.history['val_accuracy']
        plt.figure(2)
        plt.title('Linear graphs, classes :'+ str(names))
        plt.ylim(0.0, y_upper)
        plt.plot(np.linspace(0.0, len(training_loss), len(training_loss)), training_loss,'--', color = colors[n_iter-n_min_loop], label = 'train loss for ' + str(n_iter) + ' nodes ' + str(round(training_loss[-1],2)))
        plt.plot(np.linspace(0.0, len(test_accuracy), len(test_accuracy)), test_accuracy,'-', color = colors[n_iter-n_min_loop], label = 'test accuracy for ' + str(n_iter) + ' nodes ' + str(round(test_accuracy[-1],2)))
        plt.plot(np.linspace(0.0, len(test_loss), len(test_loss)), test_loss,'-.', color = tuple(t+0.3 for t in colors[n_iter-n_min_loop]), label = 'test loss for ' + str(n_iter) + ' nodes ' + str(round(test_loss[-1],2)))
        plt.xlabel('epochs')
        plt.ylabel('learning performance')
        plt.legend()
        np.savez(file_name +'/training_results' + str(n_iter), training_loss, test_accuracy, test_loss)

        plt.savefig(file_name+'/linear_graphs_cross_entropy_hanna_')

        grande_training_loss[:, average_iter] = history.history['loss']
        grande_test_accuracy[:, average_iter] = history.history['val_accuracy']
        grande_test_loss[:, average_iter] = history.history['val_loss']
        grande_precision[:, average_iter] = precision_score(y_test, y_pred, average=None)
        grande_recall[:, average_iter] = recall_score(y_test, y_pred, average=None)
        grande_f1[:, average_iter] = f1_score(y_test, y_pred, average=None)

    training_loss = np.average(grande_training_loss, 1)
    test_accuracy = np.average(grande_test_accuracy, 1)
    test_loss = np.average(grande_test_loss, 1)
    training_time_average = np.average(training_time, 1)
    precision = np.average(grande_precision, 1)
    recall = np.average(grande_recall, 1)
    f1 = np.average(grande_f1, 1)


    plt.figure(1)
    plt.title('Linear graphs, classes :'+ str(names))
    plt.ylim(0.0, y_upper)
    plt.plot(np.linspace(0.0, len(training_loss), len(training_loss)), training_loss,'--', color = colors[n_iter-n_min_loop], label = 'train loss for ' + str(n_iter) + ' nodes ' + str(round(training_loss[-1],2)))
    plt.plot(np.linspace(0.0, len(test_accuracy), len(test_accuracy)), test_accuracy,'-', color = colors[n_iter-n_min_loop], label = 'test accuracy for ' + str(n_iter) + ' nodes ' + str(round(test_accuracy[-1],2)))
    plt.plot(np.linspace(0.0, len(test_loss), len(test_loss)), test_loss,'-.', color = tuple(t+0.3 for t in colors[n_iter-n_min_loop]), label = 'test loss for ' + str(n_iter) + ' nodes ' + str(round(test_loss[-1],2)))
    plt.xlabel('epochs')
    plt.ylabel('learning performance')
    plt.legend()
    np.savez(file_name +'/training_results' + str(n_iter), training_loss, test_accuracy, test_loss)

    plt.savefig(file_name+'/linear_graphs_cross_entropy_hanna_' + str(average_num))
    

    L = ['n ' + str(n_iter) + '\n' + 
    'precision: ' + str(precision)+ '\n' +
    'recall: ' + str(recall) +'\n' + 
    'f1: ' + str(f1) + '\n\n\n']
    file1.writelines(L)
file1.close()

plt.figure(2)
plt.title('Training time, Hanna code with types:' + str(names)+ ', net_type ' + str(net_type))
plt.plot(np.arange(n_min_loop, n_max_loop), training_time_average)
plt.xlabel('number of nodes')
plt.ylabel('time in seconds')
plt.savefig(file_name +'/time_linear_graphs_cross_entropy_hanna' + str(average_num))
