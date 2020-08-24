# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 18:53:11 2020

@author: hanna
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from graph_simulation import *


class Corpus_n(object):
    '''
    Builds the corpus with the help of the Graph class.
    The size of the adjecency graphs is constant, n_max, and nodes can be unconnected.
    
    The adjecency matrices can be extracted once the corpus is built by using: corpus.corpus_list[i].A .
    The label of quantum advantage or not, can be extracted once the corpus is built by using: corpus.corpus_list[i].label 
    where [classical_advantage, quantum_advantage].
    '''
    def __init__(self, n_max, initial = 0, target = 1):
        self.corpus_list = []
        self.n_max = n_max
        self.initial = initial
        self.target = target
        self.q_count = 0
    
    def gen_color_map(self, label, G):
        # G is a networkx object
        color_map = []
        if label[1] == 1.0: # quantum
            for node in G:
                if node == self.initial:
                    color_map.append('yellow')
                elif node == self.target:
                    color_map.append('red')
                else:
                    color_map.append('magenta')
        elif label[0] == 1.0: # classical
            for node in G:
                if node == self.initial:
                    color_map.append('yellow')
                elif node == self.target:
                    color_map.append('red')
                else:
                    color_map.append('grey')
        return color_map
        
    def networkx_G_from_A(A):
        rows, cols = np.where(A == 1)
        edges = zip(rows.tolist(), cols.tolist())
        G = nx.Graph()
        G.add_edges_from(edges)
        return G

    def plot_random_graph_corpus(self):
        '''
        Will only work for random graphs.
        '''
        N = len(self.corpus_list)
        fig, axes = plt.subplots(nrows=1, ncols=N)
        ax = axes.flatten()
        plt.title('magenta = quantum, grey = classical, initial = yellow, target = red')
        
        for i in range(N):
            G = Corpus_n.networkx_G_from_A(self.corpus_list[i].A)
            color_map = self.gen_color_map(self.corpus_list[i].label, G)
            nx.draw_networkx(G, ax=ax[i], node_color=color_map, with_labels=True)
            ax[i].set_axis_off()
                
            plt.savefig('colored_linear_graphs')
        plt.figure(N+2+self.n_max)
        
        plt.title('Classical and quantum random walk in random graphs with n=' + str(self.n_max))
        plt.xlabel('time')
        plt.ylabel('Probability in target node')
        # quantum
        for i in range(N):
            plt.plot(self.corpus_list[i].t, self.corpus_list[i].pq, '-', color = (i/N, 0.2, 0.2), label = str(i) + str(not self.corpus_list[i].label[0]) +' pq, quantum')
        
        # classical
        for i in range(N):
            plt.plot(self.corpus_list[i].t, self.corpus_list[i].pc, '--', color = (i/N, 0.2, 0.2), label = str(i) + str(not self.corpus_list[i].label[1]) + ' pc, classical')
        plt.plot(np.linspace(0,10,10), np.ones(10)*self.corpus_list[i].pth, '-', color = (0, 0, 0), label = 'pth')
        plt.legend()
        plt.savefig('probability_in_t_node')    
    
    def plot_linear_graph_corpus(self):
        '''
        Does not work properly, I gave up.
        Will only work for linear graphs and not work with ghost nodes.
        '''
        N = len(self.corpus_list)
        fig, axes = plt.subplots(nrows=1, ncols=N)
        ax = axes.flatten()
        #plt.title('magenta = quantum, grey = classical, initial = yellow, target = red')
        
        for i in range(N):
            # draw from adjecency matrix because nx.draw_networkx() is weird
            gr = nx.Graph()
            for e in range(self.n_max-1):
                gr.add_edge(e, e+1)

            labels = {}

            for node in range(self.n_max):
                labels[node] = str(self.corpus_list[i].node_list[node])
            color_map = self.gen_color_map(self.corpus_list[i].label, self.corpus_list[i].node_list)
            
            nx.draw_networkx(gr, ax=ax[i], node_color=color_map, labels=labels)
            ax[i].set_axis_off()
            
        plt.savefig('colored_linear_graphs' + str(self.n_max))
        plt.figure(N+2+self.n_max)
        
        plt.title('Classical and quantum random walk in linear graphs with n=' + str(self.n_max))
        plt.xlabel('time')
        plt.ylabel('Probability in target node')
        # quantum
        for i in range(N):
            plt.plot(self.corpus_list[i].t, self.corpus_list[i].pq, '-', color = (i/N, 0.2, 0.2), label = str(i) + str(not self.corpus_list[i].label[0]) +' pq')
        
        # classical
        for i in range(N):
            plt.plot(self.corpus_list[i].t, self.corpus_list[i].pc, '--', color = (i/N, 0.2, 0.2), label = str(i) + str(not self.corpus_list[i].label[1]) + ' pc')
        plt.plot(np.linspace(0,10,10), np.ones(10)*self.corpus_list[i].pth, '-', color = (0, 0, 0), label = 'pth')
        plt.legend()
        plt.savefig('probability_in_t_node' + str(self.n_max))

    def random_graph(self, n):
        '''
        Returns [adjecency matrix in the form of a numpy array, nodes in a list]
        '''
        top_edges = (n**2-n)/2 # from Melnikov 2019 p.6
        m = random.randint(n-1, top_edges) # number of edges
        G = nx.gnm_random_graph(n, m)
        A = nx.to_numpy_matrix(G)
        return [A, G.nodes()]

    def linear_graph(self, n, all_graphs = False):
        '''
        Returns either a list of all possible graphs or one random.
        Each element in the list has [adjecency matrix in the form of a numpy array, nodes in a list]
        '''
        graph_list = self.gen_linear_graph_lists(n)
        return_list = []
        if not all_graphs:
            graph_list = [random.choice(graph_list)]
        for g in graph_list:
            A = np.zeros((n,n))
            for i in range(n-1):
                A[g[i], g[i+1]] = 1
                A[g[i+1], g[i]] = 1
            return_list.append([A, g])
        r = return_list if all_graphs else return_list[0]
        return r

    def cyclic_graph(self, n, all_graphs = False):
        '''
        Returns either a list of all possible graphs or one random.
        Each element in the list has [adjecency matrix in the form of a numpy array, nodes in a list]
        '''
        r = self.linear_graph(n, all_graphs = all_graphs)
        if all_graphs:
            for A in r:
                A[0][g[0], g[n-1]] = 1
                A[0][g[n-1], g[0]] = 1
        else:
            r[0][g[0], g[n-1]] = 1
            r[0][g[n-1], g[0]] = 1
        return r

    def gen_linear_graph_lists(self, n):
        '''
        Used in linear_graph() and cyclic_graph().
        Generate lists for making the linear graphs.
        Each list is n long, placed in an outer list.
    
        Returns every possible permutation of where self.initial and self.target can be placed so that it does not contain mirror symmetries.
        '''
        grande_list = []
        node_list = ['0'] * n # make an empty list of the right length
        # first place initial
        # can only be in ciel(n/2) places
        cont = True
        for i in range(int(np.ceil(n/2))):
            node_list[i] = self.initial
            # t has n-1 places left to be in
            for t in range(n):
                # except if i is in the middle then t can't be on both sides of i to avoid mirror symmetry
                cont = False if t > int(np.floor(n/2)) and n % 2 and i == int(np.floor(n/2)) else True
                if t != i and cont:
                    node_list[t] = self.target
                    # fill the rest of the elements with the rest of the numbers
                    l = 0
                    rest = [x for x in range(0, n) if x != self.target and x != self.initial]
                    for k in range(0, n):
                        if k != t and k != i:
                            node_list[k] = rest[l]
                            l += 1
                    # add and restart
                    grande_list.append(node_list)
                    node_list = ['0'] * n
                    node_list[i] = self.initial
        return grande_list

    def generate_graphs(self, n, N = 10, verbose = False, percentage = False, random = True, linear = False, all_graphs = False, cyclic = False):
        '''
        Not setting a percentage is faster and will lead to 15% quantum in the random case.

        Args: 
                n : number of connected nodes
                self.n_max - n : number of unconnected nodes / ghost nodes
                N : number of graphs
                verbose : 1  will draw the graphs
        '''
        
        if all_graphs:
            random = False
            if linear:
                graph_list = self.linear_graph(n, all_graphs)
            elif cyclic:
                graph_list = self.cyclic_graph(n, all_graphs)
            for graph in graph_list:
                self.corpus_list.append(GraphSimulation(graph, initial = self.initial, target = self.target))
            for l in self.corpus_list:
                if l.label[1] == 1.0:
                	self.q_count += 1

        else:
            if not percentage:
                i = 0
                while i < N:
                    if random:
                        graph = self.random_graph(n)
                    if linear:
                        graph = self.linear_graph(n)
                    if cyclic:
                        graph = self.cyclic_graph(n)
                    
                    # adding ghost nodes not connected
                    for ghost_node in range(self.n_max - n):
                        A = np.c_[A, np.zeros((n+ghost_node, 1))]
                        A = np.r_[A, np.zeros((1, n+ghost_node+1))]
                    
                    gcat = GraphSimulation(graph, initial = self.initial, target = self.target)
                    
                    
                    if not (gcat.label == np.array([0.0, 0.0])).all(): # throw away tie
                        self.corpus_list.append(gcat)
                        
                        if i % 10 == 0:
                            print('Progress in categorisation of graphs: ', i/N, '\n')
                    else:
                        print('any ties:', (gcat.label == np.array([0.0, 0.0])).all())
                        i += 1
                    i += 1

            else:
                save_list = []
                save_list_q = []
                while len(self.corpus_list) < N:
                    if random:
                        graph = self.random_graph(n)
                    if linear:
                        graph = self.linear_graph(n)
                    if cyclic:
                        graph = self.cyclic_graph(n)
                    
                    # adding ghost nodes not connected
                    for ghost_node in range(self.n_max - n):
                        A = np.c_[A, np.zeros((n+ghost_node, 1))]
                        A = np.r_[A, np.zeros((1, n+ghost_node+1))]
                    
                    # categorise
                    gcat = GraphSimulation(graph, initial = self.initial, target = self.target)
                    
                    if not (gcat.label == np.array([0.0, 0.0])).all(): # throw away tie
                        if gcat.label[1] > 0: # if quantum, gcat.label = [classical_advantage, quantum_advantage]
                            if self.q_count/N < percentage:
                                self.q_count += 1
                                self.corpus_list.append(gcat)
                            else:
                                save_list_q.append(gcat)
                                try:
                                    self.corpus_list.append(save_list.pop(0))
                                except:
                                    pass
                        else: # else classical
                            if self.q_count/N < percentage:
                                save_list.append(gcat)
                                try:
                                    self.corpus_list.append(save_list_q.pop(0))
                                    self.q_count += 1
                                except:
                                    pass
                            else:
                                self.corpus_list.append(gcat)
                    else:
                        print('ties? ', (gcat.label == np.array([0.0, 0.0])).all())
                    
                    i = len(self.corpus_list)
                    if i % 10 == 0:
                        print('Progress in categorisation of graphs: ', i/N, '\n')
                
                print('discarded classical graphs in %: ', 100*len(save_list)/N)
                print('discarded quantum graphs in %: ', 100*len(save_list_q)/N)

            print(graph)
        
'''
# check
import sys
np.set_printoptions(threshold=sys.maxsize)
N = 1
ntest = 50
corpus = Corpus_n(n_max = ntest, target = 1, initial = 0)
corpus.generate_graphs(n=ntest, N = N, linear = True, all_graphs = False)
print('q count: ', corpus.q_count/N)

c = 0
q = 0
t = 0
for l in corpus.corpus_list:
    print()
    if l.label[0] == 1.0:
        c += 1
    elif l.label[1] == 1.0:
        q += 1
    else:
        t += 1
print('c, q, t: ', c, q, t)

plt.show()
'''