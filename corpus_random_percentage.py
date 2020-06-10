# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 18:53:11 2020

@author: hanna
"""

'''
update: - size of adjecency matrix not the same as nodes in the graph, empty nodes
        - controll percentage of q nodes
        - only random graphs
'''

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy.linalg import expm
import random


class Corpus_n(object):
    '''
    Builds the corpus with the help of the Graph class.
    The size of the adjecency graphs is constant, n_max, and nodes can be unconnected.
    
    The adjecency matrices can be extracted once the corpus is built by using: corpus.corpus_list[i].A .
    The label of quantum advantage or not, can be extracted once the corpus is built by using: corpus.corpus_list[i].label .
    '''
    def __init__(self, n_max, initial = 0, target = 1):
        self.corpus_list = []
        self.n_max = n_max
        self.initial = initial
        self.target = target
        self.q_count = 0
        
    
    def plot_results(self, n, uncategorized_G):
        N = len(uncategorized_G)
        fig, axes = plt.subplots(nrows=1, ncols=N)
        ax = axes.flatten()
        for i in range(N):
            nx.draw_networkx(uncategorized_G[i], ax=ax[i], with_labels=True)
            ax[i].set_axis_off()
           
        plt.figure(N+2+self.n_max)
        
        # quantum
        for i in range(N):
            plt.plot(self.corpus_list[i].t, self.corpus_list[i].pq, '-', color = (i/N, 0.2, 0.2), label = str(i) + str(not self.corpus_list[i].label[0]) +' pq, quantum')
        
        # classical
        for i in range(N):
            plt.plot(self.corpus_list[i].t, self.corpus_list[i].pc, '--', color = (i/N, 0.2, 0.2), label = str(i) + str(not self.corpus_list[i].label[1]) + ' pc, classical')
        plt.plot(self.corpus_list[0].t, np.ones(self.corpus_list[0].t_steps)*self.corpus_list[i].pth, '-', color = (0, 0, 0), label = 'pth')
        plt.legend()
        

    def generate_random_graphs(self, n, N = 10, verbose = False, percentage = False):
        '''
        Args: 
                n : number of connected nodes
                self.n_max - n : number of unconnected nodes / ghost nodes
                N : number of graphs
                verbose : 1  will draw the graphs
        '''
        
        if not percentage:
            for i in range(N):
                top_edges = (n**2-n)/2 # from Melnikov 2019 p.6
                m = random.randint(n-1, top_edges) # number of edges
                G = nx.gnm_random_graph(n, m)
                for ghost_node in range(n, self.n_max): # add the ghost nodes
                    G.add_node(ghost_node)
                    
                if verbose:
                    plt.figure(i)
                    nx.draw_networkx(G, with_labels=True)
                
                gcat = GraphSimulation(G)
                
                self.corpus_list.append(gcat)
                
                if i % 10 == 0:
                    print('Progress in categorisation of graphs: ', i/N, '\n')

        else:
            for i in range(round(N/10)):
                # Build the graphs
                top_edges = (n**2-n)/2 # from Melnikov 2019 p.6
                m = random.randint(n-1, top_edges) # number of edges
                G = nx.gnm_random_graph(n, m)
                for ghost_node in range(n, self.n_max): # add the ghost nodes
                    G.add_node(ghost_node)
                    
                # categorise
                gcat = GraphSimulation(G)
                if gcat.label[1] > 0:
                    self.q_count += 1
                
                self.corpus_list.append(gcat)
               
            save_list = []
            while len(self.corpus_list) < N:
    
                top_edges = (n**2-n)/2 # from Melnikov 2019 p.6
                m = random.randint(n-1, top_edges) # number of edges
                G = nx.gnm_random_graph(n, m)
                for ghost_node in range(n, self.n_max): # add the ghost nodes
                    G.add_node(ghost_node)
                    
                # categorise
                gcat = GraphSimulation(G)
                if self.q_count/N < percentage:
                    if gcat.label[1] > 0:
                        self.q_count += 1
                        self.corpus_list.append(gcat)
                    else:
                        save_list.append(gcat)
                else:
                    if gcat.label[1] > 0:
                        self.q_count += 1
                        self.corpus_list.append(gcat)
                        try:
                            self.corpus_list.append(save_list.pop(0))
                        except:
                            pass
                    else:
                        self.corpus_list.append(gcat)
                
                i = len(self.corpus_list)
                if i % 10 == 0:
                    print('Progress in categorisation of graphs: ', i/N, '\n')
            
            print('discarded in %: ', len(save_list)/N)
                

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
        self.CRW_simulation() # run until finished or max_time
        
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
'''
N = 10
corpus = Corpus_n(n_max = 10, target = 1, initial = 0)
corpus.generate_random_graphs(n=5, N = N, verbose = 1, percentage = 0.7)
print(corpus.q_count/N)
'''