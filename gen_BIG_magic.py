import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from corpus_generator import *

# Saving files
import os, inspect  # for current directory
current_file_directory = os.path.dirname(os.path.abspath(__file__))
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

n_min = 25
n_max = 26
N = 5000

time_array = np.zeros(n_max - n_min)

for n in range(n_min, n_max):
    start = time.time()
    corpus = Corpus_n(n_max = n, target = 1, initial = 0)
    corpus.generate_graphs(n = n, N = N, percentage = 0, random = True, linear = False, cyclic = False, all_graphs = False, no_ties = False, magic = True)
    print('-'*10 + ' Corpus done! ' + str(n) + ' ' + '-'*10)

    data_X = np.ones((N, n, n))
    data_labels = np.ones((N, corpus.corpus_list[0].label.shape[0]))
    data_ranking = np.ones((N, corpus.corpus_list[0].ranking.shape[0]))
    for i in range(N): 
        x = corpus.corpus_list[i].A
        data_X[i] = x # numpy array
        data_labels[i] = corpus.corpus_list[i].label # 2 dim np array, categorical
        data_ranking[i] = corpus.corpus_list[i].ranking # 2 dim np array, categorical

    data_X = data_X.reshape(N, n, n, 1) # [samples, rows, columns, channels]

    end = time.time()
    timed = end-start
    time_array[n-n_min] = timed/60
 
    np.savez(current_file_directory + '/new_magic_results/dataset/' + str(n), data_X, data_labels, data_ranking)

    #np.savez(current_file_directory + '/results_magic_corpus_ranking' + '/time_N' + str(N), time_array)

print('DONE!')