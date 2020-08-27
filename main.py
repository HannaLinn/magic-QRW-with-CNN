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


def gen_data(n_max = 10, n = 5, N = 100, test_size = 0.1, val_test_size = 0.5, percentage = False, random = True, linear = False, cyclic = False, all_graphs = False):
    corpus = Corpus_n(n_max = n_max, target = 1, initial = 0)
    corpus.generate_graphs(n = n, N = N, percentage = percentage, random = random, linear = linear, cyclic = cyclic, all_graphs = all_graphs)
    print('-'*10 + ' Corpus done! ' + '-'*10)
    
    N = len(corpus.corpus_list)
    data_X = np.ones((N, n, n))
    data_labels = np.ones((N,2))
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
        X_train, y_train, X_test, y_test, X_val, y_val = load_data('train_val_test_data.npz')
    elif input('big run, random, N5000, n25, 50/50? y/n ') == 'y':
        val_test_size = 0.0
        X_train, y_train, X_test, y_test, X_val, y_val = load_data('train_val_test_dataN5000n_5050random.npz')
    else:
        val_test_size = 0.0
        X_train, y_train, X_test, y_test, X_val, y_val = load_data('train_val_test_data.npz')
    
    print('\nTraining spec.:')
    print('N_train: ', X_train.shape[0])
    q_count_train = (y_train[:,1] == 1.0).sum()
    print('q_percentage: %.2f' % (q_count_train/X_train.shape[0]))
    
    if not val_test_size == 0.0:
        print('\nValidation spec.:')
        print('N_val: ', X_val.shape[0])
        q_count_val = (y_val[:,1] == 1.0).sum()
        print('q_percentage: %.2f' %(q_count_val/X_val.shape[0]))
    
    print('\nTest spec.:')
    print('N_test: ', X_test.shape[0])
    q_count_test = (y_test[:,1] == 1.0).sum()
    print('q_percentage: %.2f' %(q_count_test/X_test.shape[0]))
    

    return X_train, y_train, X_test, y_test, X_val, y_val


X_train, y_train, X_test, y_test, X_val, y_val = get_data()


num_classes = 2

if input('train? y/n ') =='y':
    y_upper = 1.0
    batch_size = int(input('batch size (has to be a multiple of how many train and test samples), ex 3: '))
    assert len(X_train) % batch_size == 0
    assert len(X_test) % batch_size == 0
    epochs = int(input('epochs: '))
    colors = [(0.1, 0.1, 0.1), (0.9, 0.1, 0.4), (0.3, 0.2, 0.7), (0.2, 0.8, 0.6), (0.2, 0.6, 0.9)]
    
    validation_freq = 1
    n = X_train.shape[1]
    '''
    Build : batch_size, ETE_ETV_layer = True, trainable_ETE_ETV = True, num_ETE = 2, num_neurons = 10, reg_lambdas = (0.0, 0.0), con_norm = 1., dropout_rate = 0.0
    Hyper parameters:
    model_list = [(ETE_ETV_layer, trainable_ETE_ETV, reg_lambdas = (l1, l2), con_norm, dropout_rate)]

    
    model_list = [(0),
                    (1, True, (0.15, 0.45), 1.),
                    (2, True, (0.15, 0.45), 1.),
                    (3, True),
                    (False)]
    '''
    model_list = [(1, False, (0.15, 0.45), 1., 0.2)]

    for i in range(len(model_list)):
        model4 = ETE_ETV_Net(n, num_classes)
        model4 = model4.build(batch_size)
        start = time.time()
        history4 = model4.fit(X_train, y_train, batch_size=batch_size, validation_data = (X_test, y_test), validation_freq = validation_freq, epochs=epochs, verbose=2)
        end = time.time()
        vtime4 = end-start

        plt.figure(10)
        plt.title('All')
        plt.ylim(0.0, y_upper)
        plt.plot(np.linspace(0.0, epochs, epochs), history4.history['val_loss'],'--', color = colors[i], label = 'test loss for ' + str(n) + ' nodes')
        plt.plot(np.linspace(0.0, epochs, epochs), history4.history['val_accuracy'],'-', color = colors[i], label = 'test accuracy for ' + str(n) + ' nodes')
        plt.xlabel('epochs')
        plt.ylabel('learning performance')
        plt.legend()
        plt.savefig('All')

        plt.figure(i)
        plt.title(str(model_list[i]) + ' took time [min]: ' + str(vtime4/60))
        plt.ylim(0.0, y_upper)
        plt.plot(np.linspace(0.0, epochs, epochs), history4.history['val_loss'],'--', color = tuple(t+0.1 for t in colors[i]), label = 'test loss')
        plt.plot(np.linspace(0.0, epochs, epochs), history4.history['loss'],':', color = colors[i], label = 'loss')
        plt.plot(np.linspace(0.0, epochs, epochs), history4.history['val_accuracy'],'-', color = tuple(t-0.1 for t in colors[i]), label = 'test accuracy')
        plt.plot(np.linspace(0.0, epochs, epochs), history4.history['accuracy'],'-.', color = colors[i], label = 'accuracy')
        plt.xlabel('epochs')
        plt.ylabel('learning performance')
        plt.legend()
        plt.savefig('model ' + str(i))


    # ------------------------------------------------
    print('-'*20, ' DONE ', '-'*20)
    plt.show()