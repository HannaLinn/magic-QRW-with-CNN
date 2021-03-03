
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 18:53:11 2020

@author: hanna

Utility functions.

TODO:
Put more stuff here.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from corpus_generator import *
from QRWCNN_arch import *
from tensorflow.keras.utils import plot_model

'''
def load_data(filename):
    file = np.load(filename)
    X_train = file['arr_0']
    y_train = file['arr_1']
    try:
        X_test = np.concatenate((file['arr_2'], file['arr_4']), axis=0)
        y_test = np.concatenate((file['arr_3'], file['arr_5']), axis=0)
        assert np.sum(file['arr_3']) == file['arr_3'].shape[0] # any ties
    except:
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3)
    return X_train, y_train, X_test, y_test
'''
def load_data(filename):
    file = np.load(filename)
    data_X = file['arr_0']
    data_labels = file['arr_1']
    try:
        data_ranking = file['arr_2']
        return data_X, data_labels, data_ranking
    except:
        return data_X, data_labels

def delete_ties(X, y):
    mask = np.sum(y, axis=1) == 1.
    print('\nNumber of ties: ', X.shape[0] - np.sum(mask))
    X = X[mask]
    y = y[mask]
    return X, y

def div_by_batch_size(X, y, batch_size):
    if not (X.shape[0] % batch_size == 0):
        size = 1
        for i in range(batch_size+2):
            if ((X.shape[0] - i) % batch_size == 0):
                size = (X.shape[0] - i)
                break
        X = X[:size, :, :, :]
        y = y[:size, :]
    return X, y

def rinse_out(X, y, label, num_rinse):
    mask1 = np.all(y == label, axis = 1)
    mask2 = np.ones(y.shape[0], dtype='bool')
    r = 0
    num = 0
    while r < num_rinse:
        mask2[num] = False if mask1[num] else True
        r += 1 if mask1[num] else 0
        num += 1
        if num >= X.shape[0]:
            print('YOU HAVE TAKEN THEM ALL!')
            break
    X = X[mask2]
    y = y[mask2]
    return X, y


def choose_types(comp_list, data_ranking):
    mask = np.zeros(data_ranking.shape[1], dtype=bool)
    mask[comp_list] = True
    data_ranking_comp = data_ranking[:, mask]
    data_label_comp = np.zeros((data_ranking.shape[0], len(comp_list)))
    for i in range(data_ranking.shape[0]):
        temp = data_ranking_comp[i, :]
        data_label_comp[i] = np.where(temp == temp.min(), 1.0, 0.0)
    return data_label_comp

def gen_data_linear(n_max = 10, n = 5, N = 100, linear = True, cyclic = False, all_graphs = False, duplicates=False, magic=False):
    corpus = Corpus_n(n_max = n_max, target = 1, initial = 0)
    corpus.generate_graphs(n = n, N = N, percentage = False, random = False, linear = linear, cyclic = cyclic, all_graphs = all_graphs, duplicates=duplicates, magic=magic)
    print('-'*10 + ' Corpus done! ' + '-'*10)
    
    N = len(corpus.corpus_list)
    data_X = np.ones((N, n, n))
    data_labels = np.ones((N, corpus.corpus_list[0].label.shape[0]))
    data_ranking = np.ones((N, corpus.corpus_list[0].ranking.shape[0]))
    for i in range(N):
        data_X[i] = corpus.corpus_list[i].A
        data_labels[i] = corpus.corpus_list[i].label # categorical, learned embedding
        data_ranking[i] = corpus.corpus_list[i].ranking # 2 dim np array, categorical

    data_X = data_X.reshape(N, n, n, 1) # [samples, rows, columns, channels]
    
    if magic:
        np.savez(current_file_directory + '/linear_corpuses_magic' +'/train_val_test_data' + str(n), data_X, data_labels, data_ranking)
    else:
        np.savez(current_file_directory + '/linear_corpuses' +'/train_val_test_data' + str(n), data_X, data_labels, data_ranking)