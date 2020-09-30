# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 20:24:09 2020

@author: hanna
"""
import numpy as np
from corpus_generator import *

import os, inspect  # for current directory
current_file_directory = os.path.dirname(os.path.abspath(__file__))
map_name = current_file_directory + '/linear_corpuses_magic'

def load_data(filename):
    file = np.load(filename)
    data_X = file['arr_0']
    data_labels = file['arr_1']
    data_ranking = file['arr_2']
    return data_X, data_labels, data_ranking

def gen_data(n_max = 10, n = 5, N = 100, linear = True, cyclic = False, all_graphs = False, duplicates=False, magic=False):
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

def rinse_out(X, y, label, num_rinse):
    mask1 = np.all(y == label, axis = 1)
    mask2 = np.ones(y.shape[0], dtype='bool')
    r = 0
    num = 0
    while r < num_rinse:
        mask2[num] = False if mask1[num] else True
        r += 1 if mask1[num] else 0
        num += 1
    print(mask2)
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

n_iter = 4
magic = True

data_X, data_labels, data_ranking = load_data(current_file_directory + '/linear_corpuses' +'/train_val_test_data'+str(n_iter)+'.npz')

label = np.array([1., 0., 0., 0., 0., 0.])

print(np.sum(y == label))

X, y = rinse_out(label, 3)


'''
comp1 = 1 # q
comp2 = 4 # T
comp_list = [comp1, comp2]

comp_list = [1, 4, 5]
mask = np.zeros(data_ranking.shape[1], dtype=bool)
mask[comp_list] = True
print('mask ', mask)
data_ranking_comp = data_ranking[:, mask]
data_label_comp = np.zeros((data_ranking.shape[0], len(comp_list)))
for i in range(data_ranking.shape[0]):
    temp = data_ranking_comp[i, :]
    data_label_comp[i] = np.where(temp == temp.min(), 1.0, 0.0)



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
    
X_train = np.eye(3, 3)
y_train = np.eye(1, 6)
X_train = np.reshape(X_train, (1, 3, 3, 1))
for i in range(3):
    X_train = np.concatenate((X_train, X_train), axis=0)
    y_train = np.concatenate((y_train, y_train), axis=0)

X_train[0] = np.zeros((3,3,1))
y_train[0] = np.zeros(6)
X_train[6] = np.zeros((3,3,1))
y_train[6] = np.zeros(6)

print(X_train[0])
print(y_train[0])

mask =  np.sum(y_train, axis=1) == 1.
print(np.sum(mask), mask.shape, mask)

#X_train = np.delete(X_train, mask, axis=0)
#y_train = np.delete(y_train, mask, axis=0)

X_train = X_train[mask]
y_train = y_train[mask]


print(X_train[0])
print(y_train[0])
print(X_train.shape)
print(y_train.shape)

'''