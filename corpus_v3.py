# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:55:14 2020

@author: hanna
"""
'''
update: one-hot-encode for quantum_advantage = True -> label = [0 1]
'''

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy.linalg import expm


class Corpus_n(object):
    '''
    Builds the corpus with the help of the Graph class.
    The number of nodes is a constant, n (because we want it that way when going into the CNN).
    '''
    def __init__(self, n, initial = 0, target = 1):
        '''
        Check how Tensor flow wants the input data.
        '''
        self.corpus_list = []
        self.n = n
        self.initial = initial
        self.target = target
        
    '''
    Generate lists for making the line graphs.
    Each list is n long, placed in an outer list.
    
    n : number of nodes
    i : initial 
    t : target 
    2, ..., n-1 : rest of the nodes 
    
    Returns every possible permutation of where i and t can be placed so that it does not contain mirror symmetries.
    '''    
    def gen_line_graph_lists(self):
        grande_list = []
        node_list = ['0'] * self.n # make an empty list of the right length
        # first place initial
        # can only be in ciel(n/2) places
        cont = True
        for i in range(int(np.ceil(self.n/2))):
            node_list[i] = self.initial
            # t has n-1 places left to be in
            for t in range(self.n):
                # except if i is in the middle then t can't be on both sides of i to avoid mirror symmetry
                cont = False if t > int(np.floor(self.n/2)) and self.n % 2 and i == int(np.floor(self.n/2)) else True
                if t != i and cont:
                    node_list[t] = self.target
                    # fill the rest of the elements with the rest of the numbers
                    l = 0
                    rest = [x for x in range(0,self.n) if x != self.target and x != self.initial]
                    for k in range(0, self.n):
                        if k != t and k != i:
                            node_list[k] = rest[l]
                            l += 1
                    # add and restart
                    grande_list.append(node_list)
                    node_list = ['0'] * self.n
                    node_list[i] = self.initial
        return grande_list
    
    def plot_line_graphs(self, uncategorized_G):
        N = len(uncategorized_G)
        fig, axes = plt.subplots(nrows=1, ncols=N)
        ax = axes.flatten()
        for i in range(N):
            nx.draw_networkx(uncategorized_G[i], ax=ax[i], with_labels=True)
            ax[i].set_axis_off()
           
            
        plt.figure(N+2+self.n)
        
        # quantum
        for i in range(N):
            plt.plot(self.corpus_list[i].t, self.corpus_list[i].pq, '-', color = (i/N, 0.2, 0.2), label = str(i) + str(not self.corpus_list[i].label[0]) +' pq, quantum')
        
        # classical
        for i in range(N):
            plt.plot(self.corpus_list[i].t, self.corpus_list[i].pc, '--', color = (i/N, 0.2, 0.2), label = str(i) + str(not self.corpus_list[i].label[1]) + ' pc, classical')
        plt.plot(self.corpus_list[0].t, np.ones(self.corpus_list[0].t_steps)*self.corpus_list[i].pth, '-', color = (0, 0, 0), label = 'pth')
        plt.legend()
        
        
    def generate_graphs(self, line = True, cyclic = False, verbose = False):
        uncategorized_G = []
        
        # Line graphs
        graph_list = self.gen_line_graph_lists()
        for g in graph_list:
            G = nx.Graph()
            G.add_nodes_from(g)
            for i in range(self.n-1):
                G.add_edge(g[i], g[i+1])
            
            if line:
                uncategorized_G.append(G)
            
            # Cyclic graphs
            if cyclic:
                C = G.copy()
                C.add_edge(g[0], g[self.n-1])
                
                # takes away some duplicates but not all.
                if C not in uncategorized_G:
                    uncategorized_G.append(C)
        
        #nx.draw_networkx(C, with_labels=True)
        self.categorise(uncategorized_G)
        
        if verbose:
            self.plot_line_graphs(uncategorized_G)
        
    '''
    Calls the simulation class, GraphSimulation, and adds the categorised graphs to the list.
    '''
    def categorise(self, uncategorized_G):
        progress = 0
        for g in uncategorized_G:
            gcat = GraphSimulation(g, t_stop = 10.0, t_steps = 100)
            
            self.corpus_list.append(gcat)
            
            progress += 1
            if progress % 10 == 0:
                print('Progress in categorisation of graphs: ', progress/len(uncategorized_G), '\n')

class GraphSimulation(object):
    '''    
    Contains methods for simulating the quantum random walk and for classical random walk.
    Init will run simulations and categorise the graph.
    
    Args:
            graph : networkx object
            time : stop (float) default 10.0, steps (integer) default 100
            initial : initial vertex for the start of random walk (integer) default 0
            target : target vertex for the finish of the quantum walk (integer) default 1
            label : numpy array of [classical_advantage, quantum_advantage] as a float
            
    '''
    
    def __init__(self, G, t_stop = 10.0, t_steps = 100, initial = 0, target = 1):
        self.G = G
        self.A = nx.to_numpy_matrix(G)
        self.n = len(G)
        self.t = np.linspace(0.0, t_stop, t_steps)
        self.t_steps = t_steps
        self.initial = initial
        self.target = target
        
        self.pc = np.zeros((self.t_steps, 1)) # classical probability at target after simulation
        self.CRW_simulation()
        
        self.pq = np.zeros((self.t_steps, 1)) # quantum probability at target after simulation
        self.QRW_simulation()
        
        self.pth = 1/np.log(self.n)
        
        self.pc_break = np.argmax(self.pc > self.pth) if np.argmax(self.pc > self.pth) > 0 else self.t_steps
        self.pq_break = np.argmax(self.pq > self.pth) if np.argmax(self.pq > self.pth) > 0 else self.t_steps
        self.label = np.array([0, 1.0]) if self.pc_break > self.pq_break else np.array([1.0, 0])
        
    def QRW_simulation(self, gamma = 1):
        Aq = np.copy(self.A)
        Aq = np.concatenate((Aq, np.zeros((self.n,1))), axis = 1)
        Aq = np.concatenate((Aq, np.zeros((1, self.n+1))), axis = 0)
        nq = self.n + 1 # size of Fock space

        # CTQW dynamics is simulated by solving the GKSL eq, with Hamiltonian H = h_bar*A^q, h_bar = 1
        H = Aq
        H = qt.Qobj(H)
        psi0 = qt.basis(nq, self.initial) # change to args later on
        rho0 = psi0 * psi0.dag() # init cond rho0 = |1><1|, but with zero indexing
        L = qt.basis(nq, self.n) * qt.basis(nq, self.target).dag() # L = |n+1><t|, but with zero indexing
        c_op = np.sqrt(gamma) * L # collapse op C = sqrt(gamma)*L
        options = qt.Options(nsteps=1500, store_states=True, atol=1e-12, rtol=1e-12)
        result = qt.mesolve(H, rho0, self.t, c_op, options = options) # solves the master eq
        
        for i in range(self.t_steps):
            temp = result.states[i].full()
            self.pq[i] = temp[self.n,self.n] #pq = rho[n+1][n+1], [n,n] in zero indexing
            
       
    def CRW_simulation(self):
        Ac = np.copy(self.A)
        Ac[:, self.target] = 0
        Ac[self.target, self.target] = 1
        T = np.copy(Ac) # transition matrix
        for col in range(self.n):
            T[:,col] = T[:,col] / np.sum(Ac[:,col])
        p0 = np.zeros((self.n,1))
        p0[self.initial] = 1
        
        for i in range(self.t_steps):
            temp1 = np.exp(-self.t[i])
            temp2 = expm(T*self.t[i])
            f = np.dot(temp2, p0)*temp1 #eq (3) Melnikov 2019
            self.pc[i] = f[self.target,0]
'''
for i in range(6,7):
    corpus = Corpus_n(n = i, target = 1, initial = 0)
    corpus.generate_graphs(line = False, cyclic = True, verbose = True)
'''