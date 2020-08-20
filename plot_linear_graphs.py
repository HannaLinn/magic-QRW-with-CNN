from corpus_generator import *
import numpy as np
import matplotlib.pyplot as plt


n_min = 3
n_max = 17
inteference = np.zeros((n_max-n_min, 1))
inteference_add = np.zeros((n_max-n_min, 1))

for n_iter in range(n_min, n_max):
    ntest = n_iter
    corpus = Corpus_n(n_max = ntest, target = 1, initial = 0)
    corpus.generate_graphs(n=ntest, linear = True, random=False, all_graphs = True)
    #corpus.generate_graphs(n=ntest, linear = False, random=True)

    print('q count: ', corpus.q_count)
    N = len(corpus.corpus_list)

    c = 0
    q = 0
    t = 0
    for l in corpus.corpus_list:
        if l.label[0] == 1.0:
            c += 1
        elif l.label[1] == 1.0:
            q += 1
        else:
            t += 1
    print('n: ', n_iter, ', classical: ', c, ', quantum: ', q, ', ties: ', t)
    #corpus.plot_linear_graph_corpus()
    #corpus.plot_random_graph_corpus()
    #input()

    '''
    for i in range(N):
        if corpus.corpus_list[i].label[1] == 1.0:
            print(corpus.corpus_list[i].node_list)
    '''
    print('-'*20, str(n_iter), '-'*20)
    inteference[n_iter-n_min] = q/len(corpus.corpus_list)
    inteference_add[n_iter-n_min] = q

plt.figure(100)
plt.plot(np.arange(n_min, n_max), inteference)
plt.xlabel('number of nodes')
plt.ylabel('percentage of quantum label')
plt.title('Percentage over number of nodes')
plt.savefig('percentage_q_over_n')

plt.figure(101)
plt.plot(np.arange(n_min, n_max), inteference_add)
plt.xlabel('number of nodes')
plt.ylabel('number of quantum label')
plt.title('Quantum labeled over number of nodes')
plt.savefig('addition_q_over_n')