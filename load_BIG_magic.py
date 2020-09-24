import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from corpus_generator import *

# Saving files
import os, inspect  # for current directory
current_file_directory = os.path.dirname(os.path.abspath(__file__))
os.environ["CUDA_VISIBLE_DEVICES"] = ""

map_name = current_file_directory + '/results_magic_corpus_ranking'

def load_data(filename):
    file = np.load(filename)
    data_X = file['arr_0']
    data_labels = file['arr_1']
    data_ranking = file['arr_2']
    return data_X, data_labels, data_ranking

print_list = []
print_list_q = []
n_min = 4
n_max = 20

count_norm_array = np.zeros((6, (n_max-n_min)))
count_q_ = np.zeros((5,(n_max-n_min)))

for i in range(n_min, n_max):
    filename = map_name + '/train_val_test_data_n_' + str(i) + '.npz'
    data_X, data_labels, data_ranking = load_data(filename)

    N = data_X.shape[0]

    num_types = data_labels[0].shape[0]
    count = np.zeros(num_types, dtype=int)
    ties = 0
    better = 0
    better_q = 0
    better_q_p = 0
    better_q_n = 0
    better_q_T = 0
    better_q_H = 0 
    better_n_p = 0

    for l in range(N):
        count[np.argmax(data_labels[l])] += 1
        ties += (np.sum(data_labels[l]) != 1)
        better_q_p += (data_ranking[l][1] < data_ranking[l][2])
        better_q_n += (data_ranking[l][1] < data_ranking[l][3])
        better_q_T += (data_ranking[l][1] < data_ranking[l][4])
        better_q_H += (data_ranking[l][1] < data_ranking[l][5])
        better_n_p += (data_ranking[l][2] < data_ranking[l][3])

    count_q_[0, i-n_min] = better_q_p/N
    count_q_[1, i-n_min] = better_q_n/N
    count_q_[2, i-n_min] = better_q_T/N
    count_q_[3, i-n_min] = better_q_H/N
    count_q_[4, i-n_min] = better_n_p/N


    ties = ties/N
    count_norm = count / N
    count_norm_array[:, i-n_min] = np.array(count_norm)
    print_list.append((str(i) + ' & ' + str(count_norm[0]) + ' & ' + str(count_norm[1]) + ' & ' + str(count_norm[2]) + ' & ' + str(count_norm[3]) + ' & ' + str(count_norm[4]) + ' & ' + str(count_norm[5]) +' & ' + str(ties) + '\\\\ \\hline'))

'''
for p in print_list:
    print(p)
'''

colors = ['grey', 'magenta', 'cyan', 'yellow', 'red', 'blue', 'orange']
names = ['c', 'q', 'pos', 'neg', 'T', 'H', 'neg<pos']

for j in range(0,6):
    plt.figure(1)
    plt.plot(np.arange(n_min, n_max), count_norm_array[j, :], label = names[j], color = colors[j])
    plt.xlabel('n')
    plt.ylabel('percentage of labels (div by N)')
    plt.title('How often magic is better than q')
    plt.legend()
    plt.savefig('Magic_better_N')

for j in range(0,5):
    plt.figure(2)
    plt.plot(np.arange(n_min, n_max), count_q_[j, :], label = names[j+2], color = colors[j+2])
    plt.xlabel('n')
    plt.ylabel('percentage of labels better (div by N)')
    plt.title('How often magic is better than q (ties not included)')
    plt.legend()
    plt.savefig('Magic_better_q')