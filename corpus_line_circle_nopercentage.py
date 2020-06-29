# -*- coding: utf-8 -*-
"""
Created on Tue May 26 10:36:09 2020

@author: hanna
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy.linalg import expm
import random


class Corpus_n(object):
    '''
    Builds the corpus with the help of the Graph class.
    The number of nodes is a constant, n, and decides the size of the adjecency matrix.
    But nodes can be unconnected.
    
    The adjecency matrices can be extracted once the corpus is built by using: corpus.corpus_list[i].A .
    The label of quantum advantage or not, can be extracted once the corpus is built by using: corpus.corpus_list[i].label .
    '''
    def __init__(self, n, initial = 0, target = 1):
        self.corpus_list = []
        self.n = n
        self.initial = initial
        self.target = target
            
    def gen_line_graph_lists(self):
            '''
        Used for generating line graphs.
        Generate lists for making the line graphs.
        Each list is n long, placed in an outer list.
    
        Returns every possible permutation of where i and t can be placed so that it does not contain mirror symmetries.
        '''
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
    
    def plot_results(self, uncategorized_G):
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
        

    def generate_random_graphs(self, N = 10, regular = False, total_random = False, verbose = False):
        uncategorized_G = []
        '''
        Args: 
                N : number of graphs
                regular : will generate random regular graphs
                total_random : will generate totally random graphs 
                verbose : 1 will run function plot_results, 2 will draw the graphs
        '''
        
        for i in range(N):
            if total_random:
                top_edges = (self.n**2-self.n)/2 # from Melnikov 2019 p.6
                m = random.randint(self.n-1, top_edges)
                G = nx.gnm_random_graph(self.n, m)
                uncategorized_G.append(G)
            if regular:
                # n * d must be even
                d_list = [x for x in range(1, self.n) if (x*self.n) % 2 == 0]
                d = random.choice(d_list)
                G = nx.random_regular_graph(d, self.n)
                uncategorized_G.append(G)
                
            if verbose == 2:
                plt.figure(i)
                nx.draw_networkx(G, with_labels=True)
        self.categorise(uncategorized_G)
        
        if verbose:
            self.plot_results(uncategorized_G)
    

    def generate_graphs(self, line = False, cyclic = False, verbose = False):
        uncategorized_G = []
        
        '''
        Args: 
                line : makes line graphs
                cyclic : makes cyclic graphs
                verbose : 1 will run function plot_results
        '''
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
        
        if verbose:
            self.plot_results(uncategorized_G)
        

    def categorise(self, uncategorized_G):
        '''
        Calls the simulation class, GraphSimulation, and adds the categorised graphs to the list.
        '''
        progress = 0
        for g in uncategorized_G:
            gcat = GraphSimulation(g)
            
            self.corpus_list.append(gcat)
            
            progress += 1
            if progress % 10 == 0:
                print('Progress in categorisation of graphs: ', progress/len(uncategorized_G), '\n')



class GraphSimulation(object):
    '''    
    Contains methods for simulating the quantum random walk and for classical random walk.
    Init will run simulations and categorise the graph.
    
    Args:
            G = graph : networkx object
            initial : initial vertex for the start of random walk (integer) default 0
            target : target vertex for the finish of the quantum walk (integer) default 1
            A : is a nxn numpy array of the adjecency matrix
            label : numpy array of [classical_advantage, quantum_advantage] as a float
            
    '''
    def __init__(self, G, step_size = 0.10, initial = 0, target = 1):
        self.G = G
        self.A = nx.to_numpy_matrix(G)
        self.n = len(G)
        self.step_size = step_size
        self.max_time = step_size * 1000 * self.n
        self.initial = initial
        self.target = target
        self.pth = 1/np.log(self.n)
        
        self.pc_hitting_time = 0
        self.pc = np.zeros((1,1)) # classical probability at target after simulation
        self.CRW_simulation() # run until finished
        
        self.pq_hitting_time = 100
        self.t_steps = int(np.ceil(self.pc_hitting_time / self.step_size))
        self.t = np.linspace(0.0, self.pc_hitting_time, self.t_steps)
        self.pq = np.zeros((self.t_steps, 1)) # quantum probability at target after simulation
        self.QRW_simulation()
        
        self.label = np.array([0, 1.0]) if self.pc_hitting_time > self.pq_hitting_time else np.array([1.0, 0])
        
        
    def QRW_simulation(self, gamma = 1):
        Aq = np.copy(self.A)
        Aq = np.concatenate((Aq, np.zeros((self.n,1))), axis = 1)
        Aq = np.concatenate((Aq, np.zeros((1, self.n+1))), axis = 0)
        nq = self.n + 1 # size of Fock space

        # CTQW dynamics is simulated by solving the GKSL eq, with Hamiltonian H = h_bar*A^q, h_bar=1
        H = Aq
        H = qt.Qobj(H)
        psi0 = qt.basis(nq, self.initial) # change to args later on
        rho0 = psi0 * psi0.dag() # init cond rho0 = |1><1|, but with zero indexing
        L = qt.basis(nq, self.n) * qt.basis(nq, self.target).dag() # L = |n+1><t|, but with zero indexing
        c_op = np.sqrt(gamma) * L # collapse op C = sqrt(gamma)*L
        options = qt.Options(nsteps=1500, store_states=True, atol=1e-12, rtol=1e-12)
        result = qt.mesolve(H, rho0, self.t, c_op, options = options) # solves the master eq
        
        found = False
        for i in range(self.t_steps):
            temp = result.states[i].full()
            self.pq[i] = temp[self.n,self.n].real #pq = rho[n+1][n+1], [n,n] in zero indexing
            if self.pq[i] > self.pth and not found:
                self.pq_hitting_time = self.pq[i]
                found = True
                
       
    def CRW_simulation(self):
        Ac = np.copy(self.A)
        Ac[:, self.target] = 0
        Ac[self.target, self.target] = 1
        T = np.copy(Ac) # transition matrix
        for col in range(self.n):
            T[:,col] = T[:,col] / np.sum(Ac[:,col]) if np.sum(Ac[:,col]) != 0 else np.zeros(self.n) # div by 0!!!
        p0 = np.zeros((self.n,1))
        p0[self.initial] = 1
        
        t = self.step_size # don't need to calculate t = 0 as that is 0.0
        prob = 0
        while prob < self.pth and t < self.max_time:
            temp1 = np.exp(-t)
            temp2 = expm(T*t)
            f = np.dot(temp2, p0)*temp1 #eq (3) Melnikov 1
            prob = f[self.target,0]
            self.pc = np.append(self.pc, prob)
            t = round(t + self.step_size, 6)
        self.pc_hitting_time = t

#corpus = Corpus_n(n = 25, target = 1, initial = 0)
#corpus.generate_random_graphs(N = 20, total_random=True, verbose = 2)
        