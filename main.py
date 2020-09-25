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


# Saving files
import os, inspect  # for current directory
current_file_directory = os.path.dirname(os.path.abspath(__file__))
from sklearn.metrics import f1_score, precision_score, recall_score

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tf.config.list_physical_devices('GPU')

file_dir = current_file_directory + '/results_main'


def gen_data(n_max = 10, n = 5, N = 100, test_size = 0.1, val_test_size = 0.5, percentage = False, random = True, linear = False, cyclic = False, all_graphs = False, num_classes = 2):
    corpus = Corpus_n(n_max = n_max, target = 1, initial = 0)
    corpus.generate_graphs(n = n, N = N, percentage = percentage, random = random, linear = linear, cyclic = cyclic, all_graphs = all_graphs, no_ties = True)
    print('-'*10 + ' Corpus done! ' + '-'*10)
    
    N = len(corpus.corpus_list)
    data_X = np.ones((N, n, n))
    data_labels = np.ones((N, num_classes))
    for i in range(N): 
        x = corpus.corpus_list[i].A
        data_X[i] = x # numpy array
        data_labels[i] = corpus.corpus_list[i].label # 2 dim np array, categorical

    data_X = data_X.reshape(N, n, n, 1) # [samples, rows, columns, channels]
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_labels, test_size=test_size)
    
    if not val_test_size == 0.0:
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=val_test_size)
    else:
        X_test, X_val, y_test, y_val = X_test, np.zeros((1,1)), y_test, np.zeros((1,1))
    np.savez(file_dir + '/train_val_test_data', X_train, y_train, X_test, y_test, X_val, y_val)

def load_data(filename):
    file = np.load(filename)
    X_train = file['arr_0']
    y_train = file['arr_1']
    try:
        X_test = file['arr_2']
        y_test = file['arr_3']
        X_val = file['arr_4']
        y_val = file['arr_5']
    except:
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3)
        X_test, X_val, y_test, y_val = X_test, np.zeros((1,1)), y_test, np.zeros((1,1))
    return X_train, y_train, X_test, y_test, X_val, y_val


def get_data():
    if input('reload corpus? y/n ') == 'y':
        print('\nTest data: independent data on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data.')
        test_size = float(input('test size (for ex 0.1): '))
        
        print('\nValidation data: independent data on which to evaluate after training.')
        val_test_size = float(input('validation size, split from test data, if not used set to 0.0 (for ex 0.1): '))
        s = input('random, linear or cyclic? r/l/c ')
        if s == 'r':
            N = int(input('N: '))
            n_max = int(input('n_max: '))
            n = int(input('n (lower or eq. to n_max): '))
            percentage = float(input('percentage of q in corpus: '))
            gen_data(n_max = n_max, n = n, N = N, test_size = test_size, val_test_size = val_test_size, percentage = percentage)
        elif s == 'l':
            all_graphs = int(input('all possible graphs? 1/0 '))
            N = 10 # place holders
            percentage = 0.1 # place holders
            if not all_graphs:
                N = int(input('N: '))
                percentage = float(input('percentage of q in corpus: '))
            n_max = int(input('n_max: '))
            n = int(input('n (lower or eq. to n_max): '))
            gen_data(n_max = n_max, n = n, N = N, test_size = test_size, val_test_size = val_test_size, percentage = percentage, random = False, linear = True, all_graphs = all_graphs)
        elif s == 'c':
            all_graphs = int(input('all possible graphs? 1/0 '))

            if not all_graphs:
                N = int(input('N: '))
                percentage = float(input('percentage of q in corpus: '))
            n_max = int(input('n_max: '))
            n = int(input('n (lower or eq. to n_max): '))
            gen_data(n_max = n_max, n = n, N = N, test_size = test_size, val_test_size = val_test_size, percentage = percentage, random = False, cyclic = True, all_graphs = all_graphs)
        else:
            raise NameError('Type not correctly given, try again.')
        X_train, y_train, X_test, y_test, X_val, y_val = load_data(file_dir + '/train_val_test_data.npz')
    elif input('big run, random, N5000, n10, 50/50? y/n ') == 'y':
        val_test_size = 0.0
        X_train, y_train, X_test, y_test, X_val, y_val = load_data(file_dir + '/train_val_test_data_big.npz')
    elif input('BIG run, random, N10000, 50/50? y/n ') == 'y':
        n = int(input('n: '))
        val_test_size = 0.0
        X_train, y_train, X_test, y_test, X_val, y_val = load_data( current_file_directory + '/results_nomagic_corpus' + '/train_val_test_data_n_' + str(n) + '.npz')    
    elif input('MAGIC BIG run, random, N10000, 50/50? y/n ') == 'y':
        n = int(input('n: '))
        val_test_size = 0.0
        X_train, y_train, X_test, y_test, X_val, y_val = load_data( current_file_directory + '/results_magic_corpus' + '/train_val_test_data_n_' + str(n) + '.npz')    
    else:
        val_test_size = 0.0
        X_train, y_train, X_test, y_test, X_val, y_val = load_data(file_dir + '/train_val_test_data.npz')
    


    return X_train, y_train, X_test, y_test, X_val, y_val

def delete_ties(X, y, batch_size):
    mask = np.sum(y, axis=1) == 1.
    print('\nNumber of ties: ', X.shape[0] - np.sum(mask))
    X = X[mask]
    y = y[mask]
    if not (X.shape[0] % batch_size == 0):
        size = 1
        for i in range(batch_size+2):
            if ((X.shape[0] - i) % batch_size == 0):
                size = (X.shape[0] - i)
                break
        X = X[:size, :, :, :]

    if not (y.shape[0] % batch_size == 0):
        size = 1
        for i in range(batch_size+2):
            if ((y.shape[0] - i) % batch_size == 0):
                size = (y.shape[0] - i)
                break
        y = y[:size]

    return X, y

X_train, y_train, X_test, y_test, X_val, y_val = get_data()


num_classes = y_test.shape[1]
if num_classes > 2:
    file_dir = current_file_directory + '/results_main_magic'

if input('train? y/n ') =='y':

    batch_size = int(input('\nbatch size (has to be a multiple of how many train and test samples), ex 3: '))
    epochs = int(input('epochs: '))

    print('\nN_train with ties: ', X_train.shape[0])
    print('N_test with ties: ', X_test.shape[0])
    X_train, y_train = delete_ties(X_train, y_train, batch_size)
    X_test, y_test = delete_ties(X_test, y_test, batch_size)
    X_val, y_val = delete_ties(X_val, y_val, batch_size)

    print('(N, n, n, C): ', X_train.shape)

    print('\nTraining spec.:')
    print('N_train: ', X_train.shape[0])
    q_count_train = (y_train[:,1] == 1.0).sum()
    print('q_percentage: %.2f' % (q_count_train/X_train.shape[0]))
    
    if not np.all(X_val == np.zeros((1,1))):
        print('\nValidation spec.:')
        print('N_val: ', X_val.shape[0])
        q_count_val = (y_val[:,1] == 1.0).sum()
        print('q_percentage: %.2f' %(q_count_val/X_val.shape[0]))
    
    print('\nTest spec.:')
    print('N_test: ', X_test.shape[0])
    q_count_test = (y_test[:,1] == 1.0).sum()
    print('q_percentage: %.2f' %(q_count_test/X_test.shape[0]))

    assert not np.any(np.isnan(y_test))
    assert np.sum(y_test) == y_test.shape[0] # any ties

    y_upper = 1.0
    
    assert len(X_train) % batch_size == 0
    assert len(X_test) % batch_size == 0
    assert np.all(np.sum(y_train, axis = 1) == 1.) # no ties
    
    colors = [(0.1, 0.1, 0.1), (0.9, 0.1, 0.4), (0.3, 0.2, 0.7), (0.2, 0.8, 0.6), (0.2, 0.6, 0.9),
                (0.2, 0.1, 0.1), (0.8, 0.1, 0.4), (0.4, 0.2, 0.7), (0.3, 0.8, 0.6), (0.3, 0.6, 0.9)]
    
    validation_freq = 1
    n = X_train.shape[1]

    file1 = open(file_dir +'/training_param.txt', 'w')
    q_count_train = (y_train[:,1] == 1.0).sum()
    q_count_test = (y_test[:,1] == 1.0).sum()
    L = ['batch_size: ' + str(batch_size) + '\n', 'epochs: ' + str(epochs) + '\n', 'train (N, n, n, C): ' + str(X_train.shape)+ ' q_percentage: %.2f' % (q_count_train/X_train.shape[0]) + '\n', 'test (N, n, n, C): ' + str(X_test.shape)+ ' q_percentage: %.2f' %(q_count_test/X_test.shape[0]) + '\n']
    file1.writelines(L)
    file1.close()

    X_train, y_train, X_test, y_test, X_val, y_val = tf.convert_to_tensor(X_train), tf.convert_to_tensor(y_train), tf.convert_to_tensor(X_test), tf.convert_to_tensor(y_test), tf.convert_to_tensor(X_val), tf.convert_to_tensor(y_val)


    '''
    *Init* : n,
    num_classes = 2

    net_type = 1,
    conv_learn = False,
    num_ETE = 2,
    num_neurons = 10,
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
    model_list = [[(1, True, 2, 10), ((0.0, 0.0), 1000., 0.0)],
                    [(1, True, 3, 10), ((0.0, 0.0), 1000., 0.0)],
                    [(1, False, 2, 10), ((0.0, 0.0), 1000., 0.0)],
                    [(1, True, 2, 5), ((0.0, 0.0), 1000., 0.0)],
                    [(1, True, 2, 20), ((0.0, 0.0), 1000., 0.0)],
                    [(1, True, 2, 10), ((0.15, 0.45), 1000., 0.0)],
                    [(1, True, 2, 10), ((0.0, 0.0), 1., 0.0)],
                    [(1, True, 2, 10), ((0.0, 0.0), 1000., 0.2)],
                    [(3, True, 1, 10), ((0.0, 0.0), 1000., 0.0)]]


    file1 = open(file_dir +'/f1_precision_recall.txt', 'w')
    for i in range(len(model_list)):
        print(*model_list[i][0])
        print(*model_list[i][1])
        model = ETE_ETV_Net(n, num_classes, *model_list[i][0])
        model = model.build(batch_size, *model_list[i][1])
        plot_model(model, to_file=file_dir + 'model_plot' + str(i) + 'm.png', show_shapes=True, show_layer_names=True)
        start = time.time()
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, min_delta = 0.0001), tf.keras.callbacks.TerminateOnNaN()]
        history4 = model.fit(X_train, y_train, callbacks=callbacks, batch_size=batch_size, validation_data = (X_test, y_test), validation_freq = validation_freq, epochs=epochs, verbose=2, shuffle = True)        
        
        end = time.time()
        vtime4 = end-start

        plt.figure(10)
        plt.title('All for n:' + str(n))
        plt.ylim(0.0, y_upper)
        y = history4.history['val_loss']
        plt.plot(np.linspace(0.0, len(y), len(y)), y,'--', color = colors[i], label = 'test loss model ' + str(i))
        y = history4.history['val_accuracy']
        plt.plot(np.linspace(0.0, len(y), len(y)), y,'-', color = colors[i], label = 'test accuracy model ' + str(i))
        plt.xlabel('epochs')
        plt.ylabel('learning performance')
        plt.legend()
        plt.savefig(file_dir +'/All')

        plt.figure(i)
        plt.title(str(model_list[i]) + ' took time [min]: ' + str(round(vtime4/60,3)))
        #plt.ylim(0.0, y_upper)
        y = history4.history['val_loss']
        plt.plot(np.linspace(0.0, len(y), len(y)), y,'--', color = tuple(t+0.1 for t in colors[i]), label = 'test loss')
        y = history4.history['loss']
        plt.plot(np.linspace(0.0, len(y), len(y)), y,':', color = colors[i], label = 'loss')
        y = history4.history['val_accuracy']
        plt.plot(np.linspace(0.0, len(y), len(y)), y,'-', color = tuple(t-0.1 for t in colors[i]), label = 'test accuracy')
        y = history4.history['accuracy']
        plt.plot(np.linspace(0.0, len(y), len(y)), y,'-.', color = colors[i], label = 'accuracy')
        plt.xlabel('epochs')
        plt.ylabel('learning performance')
        plt.legend()
        plt.savefig(file_dir +'/model ' + str(i))
        np.savez(file_dir +'/training_results' + str(i), history4.history['loss'], history4.history['val_accuracy'], history4.history['val_loss'])


        y_pred1 = model.predict(X_test, batch_size=batch_size)
        y_pred=np.eye(1, num_classes, k=np.argmax(y_pred1, axis =1)[0])
        for j in range(1, y_pred1.shape[0]):
            y_pred = np.append(y_pred, np.eye(1, num_classes, k=np.argmax(y_pred1, axis =1)[j]), axis = 0)

        L = ['model ' + str(i) + '\n' + 
        'precision: ' + str(precision_score(y_test, y_pred, average=None))+ '\n' +
        'recall: ' + str(recall_score(y_test, y_pred, average=None)) +'\n' + 
        'f1: ' + str(f1_score(y_test, y_pred, average=None)) + '\n\n\n']
        file1.writelines(L)
    file1.close()

    # ------------------------------------------------
    print('-'*20, ' DONE ', '-'*20)
    plt.show()
