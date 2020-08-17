from corpus_generator import *


for n_iter in range(3, 10):
	ntest = n_iter
	corpus = Corpus_n(n_max = ntest, target = 1, initial = 0)
	corpus.generate_graphs(n=ntest, linear = True, random=False, all_graphs = True)
	print('q count: ', corpus.q_count)
	print('N ', len(corpus.corpus_list))

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
	corpus.plot_linear_graph_corpus()

print('DONE!')