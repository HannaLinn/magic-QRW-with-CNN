# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 18:57:48 2020

@author: hanna

Plot learning curves.

"""
# Saving files
import os, inspect  # for current directory

import numpy as np
import matplotlib.pyplot as plt
current_file_directory = os.path.dirname(os.path.abspath(__file__))
file_dir = current_file_directory + '/results_main_magic'
n = 7
filename = file_dir +'/train_results' + '.npz'

file = np.load(filename)
train_loss = file['arr_0']
test_accuracy = file['arr_1']
test_loss = file['arr_2']

plt.figure(2)
plt.title(str([(1, True, 2, 10, 2), ((0.0, 0.0), 1000., 0.0)]) + ' random with types ')
#plt.ylim(0.0, y_upper)
plt.plot(np.linspace(0.0, len(train_loss), len(train_loss)), train_loss,':', label = 'train loss ' + str(round(train_loss[-1],2)))
plt.plot(np.linspace(0.0, len(test_accuracy), len(test_accuracy)), test_accuracy,'-', label = 'test accuracy ' + str(round(test_accuracy[-1],2)))
#plt.plot(np.linspace(0.0, len(train_accuracy), len(train_accuracy)), train_accuracy,'-.', label = 'train accuracy ' + str(round(train_accuracy[-1],2)))
plt.plot(np.linspace(0.0, len(test_loss), len(test_loss)), test_loss,'--', label = 'test loss ' + str(round(test_loss[-1],2)))
plt.xlabel('epochs')
plt.ylabel('learning performance')
plt.legend()

plt.savefig(file_dir+'/random_graphs_')
