# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 16:29:26 2020

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



y_upper = 1.0
batch_size = 1
epochs = 1000
colors = [(51, 51, 51), (76, 32, 110), (32, 92, 25), (32, 95, 105), (110, 32, 106), (189, 129, 9)]# ['grey', 'purple', 'green', 'blue', 'magenta', 'orange']
colors = [tuple(t/250 for t in x) for x in colors]

validation_freq = 1

average_num = 10
n_min_loop = 4
n_max_loop = 8
training_time = np.zeros((n_max_loop-n_min_loop,average_num))
reg = (0.0, 0.0)
dropout_rate = 0.0
num_classes = 2


for n_iter in range(n_min_loop, n_max_loop):

    grande_training_loss = np.zeros((epochs, average_num))
    grande_test_accuracy = np.zeros((epochs, average_num))
    grande_test_loss = np.zeros((epochs, average_num))
    
    for average_iter in range(average_num):
        print('-'*20, str(average_iter), '-'*20)
        gen_data(n_max = n_iter, n = n_iter, test_size = 0.1, val_test_size = 0.0, random = False, linear = True, cyclic = False, all_graphs = True)
        X_train, y_train, X_test, y_test, X_val, y_val = load_data('train_val_test_data.npz')
        assert len(X_train) % batch_size == 0
        assert len(X_test) % batch_size == 0

        model = ETE_ETV_Net(n_iter, num_classes)
        model = model.build(batch_size)
        start = time.time()
        history = model.fit(X_train, y_train, batch_size=batch_size, validation_data = (X_test, y_test), validation_freq = validation_freq, epochs=epochs, verbose=2)
        end = time.time()
        vtime = end-start
        training_time[n_iter-n_min_loop, average_iter] = vtime

        grande_training_loss[:, average_iter] = history.history['loss']
        grande_test_accuracy[:, average_iter] = history.history['val_accuracy']
        grande_test_loss[:, average_iter] = history.history['val_loss']

    training_loss = np.average(grande_training_loss, 1)
    test_accuracy = np.average(grande_test_accuracy, 1)
    test_loss = np.average(grande_test_loss, 1)
    training_time_average = np.average(training_time, 1)

    plt.figure(1)
    plt.title('Figure 3 a) Hanna code, average '+ str(average_num) + ', dropout ' + str(dropout_rate) + ', reg ' + str(reg))
    plt.ylim(0.0, y_upper)
    plt.plot(np.linspace(0.0, len(training_loss), len(training_loss)), training_loss,'--', color = colors[n_iter-n_min_loop], label = 'train loss for ' + str(n_iter) + ' nodes')
    plt.plot(np.linspace(0.0, len(test_accuracy), len(test_accuracy)), test_accuracy,'-', color = colors[n_iter-n_min_loop], label = 'test accuracy for ' + str(n_iter) + ' nodes')
    plt.plot(np.linspace(0.0, len(test_loss), len(test_loss)), test_loss,'-.', color = tuple(t+0.3 for t in colors[n_iter-n_min_loop]), label = 'test loss for ' + str(n_iter) + ' nodes')
    plt.xlabel('epochs')
    plt.ylabel('learning performance')
    plt.legend()

plt.savefig('linear_graphs_cross_entropy_hanna_' + str(average_num))

plt.figure(2)
plt.title('Training time, Hanna code')
plt.plot(np.arange(n_min_loop, n_max_loop), training_time_average)
plt.xlabel('number of nodes')
plt.ylabel('time in seconds')
plt.savefig('time_linear_graphs_cross_entropy_hanna' + str(average_num))


# ------------------------------------------------
print('-'*20, ' DONE ', '-'*20)