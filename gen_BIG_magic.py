import numpy as np
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
import time
from corpus_generator import *

# Saving files
import os, inspect  # for current directory
current_file_directory = os.path.dirname(os.path.abspath(__file__))
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


N = 5000

def gen_data(n_max = 10, n = 5, N = 100, random = True, linear = False, cyclic = False, all_graphs = False, duplicates = False, magic=True, percentage=False):
    '''
    Build a new dataset to be stored in dataset folder.
    '''
    corpus = Corpus_n(n_max = n_max, target = 1, initial = 0)
    corpus.generate_graphs(n = n, N = N, percentage = percentage, random = random, linear = linear, cyclic = cyclic, all_graphs = all_graphs, duplicates=duplicates, magic=magic)
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
    
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    if linear:
        np.savez(current_file_directory + '/datasets' + '/linear_graph_datasets' + '/train_val_test_data_n_' + str(n), data_X, data_labels, data_ranking)
    elif cyclic:
        np.savez(current_file_directory + '/datasets' + '/cyclic_graph_datasets' + '/train_val_test_data_n_' + str(n), data_X, data_labels, data_ranking)
    elif random:
        np.savez(current_file_directory + '/datasets' + '/random_graph_datasets' + '/train_val_test_data_n_' + str(n), data_X, data_labels, data_ranking)


for n in [4,20,21,22,23,24,25]:
    gen_data(n_max = n, n = n, N = 5000, random = False, linear = True, cyclic = False, all_graphs = True, duplicates = False, magic=True, percentage=False)
    
    '''
    start = time.time()
    corpus = Corpus_n(n_max = n, target = 1, initial = 0)
    corpus.generate_graphs(n = n, N = N, percentage = 0, random = False, linear = True, cyclic = False, all_graphs = False, no_ties = False, magic = True)
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

    #end = time.time()
    #timed = end-start
    #time_array[n-n_min] = timed/60
 
    np.savez(current_file_directory + '/linear_datasets/' + str(n), data_X, data_labels, data_ranking)

    #np.savez(current_file_directory + '/results_magic_corpus_ranking' + '/time_N' + str(N), time_array)
    '''

print('DONE!')
