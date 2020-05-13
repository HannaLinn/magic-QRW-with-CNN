# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 13:54:52 2020

@author: hanna
"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy.linalg import expm

'''
Makes the example on page 3 Melnikov et al 2019
'''

class ExampleCorpus(object):
    '''
    Make the graph in example on page 3 Menlikov 2019.
    0. Make list
    1. The three graph as networkx objects
    2. Put in a list
    3. Categorise.
    One object is a list.
    '''
    
    def __init__(self):
        self.corpus_list = []
        
    def generate_graphs(self):
        uncategorized_G = []
        # graph 1
        G1 = nx.Graph()
        nodes = [0, 1, 2]
        G1.add_nodes_from(nodes, sink = False)
        edges = [(0,2), (2,1)]
        G1.add_edges_from(edges)
        uncategorized_G.append(G1)
        
        # graph 2
        G2 = nx.Graph()
        nodes = [0, 1, 2]
        G2.add_nodes_from(nodes, sink = False)
        edges = [(0,1), (1,2)]
        G2.add_edges_from(edges)
        uncategorized_G.append(G2)
        
        # graph 3
        G3 = nx.Graph()
        nodes = [0, 1, 2]
        G3.add_nodes_from(nodes, sink = False)
        edges = [(2,0), (0,1)]
        G3.add_edges_from(edges)
        uncategorized_G.append(G3)
        
        
        self.categorise(uncategorized_G)
        
    def categorise(self, uncategorized_G):
        for g in uncategorized_G:
            gcat = GraphSimulation(g)
            gcat.QRW_simulation()
            gcat.CRW_simulation()
            self.corpus_list.append(gcat)

    def plot_corpus(self):
        colors = ['blue', 'green', 'gray']
        
        # Plot the graphs
        fig, axes = plt.subplots(nrows=1, ncols=3)
        ax = axes.flatten()
        for i in range(3):
            nx.draw_networkx(self.corpus_list[i].G, ax=ax[i], node_color = colors[i], with_labels=True)
            ax[i].set_axis_off()
        
        # Plot the probabilities
        plt.figure(2)
        for i in range(3):
            plt.plot(self.corpus_list[i].t, self.corpus_list[i].pq, '-', color = colors[i], label = 'pq, quantum')

        for i in range(3):
            plt.plot(self.corpus_list[i].t, self.corpus_list[i].pc, '--', color = colors[i], label = 'pc, classical')

        pth  = np.ones(100) * 1/np.log(3) # Should not be here! Just for the plot
        plt.plot(self.corpus_list[0].t, pth, '-', color = 'black')
        plt.title('(b)')
        plt.ylabel('detection probabibilies, pq and pc')
        plt.xlabel('time, t')
        plt.legend()
        #plt.show()


class GraphSimulation(object):
    '''    
    Contains methods for simulating the quantum random walk and for classical random walk.
    
    Args:
            graph : networkx object
            time : stop (float) default 10.0, steps (integer) default 100
            initial : initial vertex for the start of random walk (integer) default 0
            target : target vertex for the finish of the quantum walk (integer) default 1
            
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
        self.pq = np.zeros((self.t_steps, 1)) # quantum probability at target after simulation
    
    def QRW_simulation(self, gamma = 1):
        Aq = np.copy(self.A)
        Aq = np.concatenate((Aq, np.zeros((self.n,1))), axis = 1)
        Aq = np.concatenate((Aq, np.zeros((1, self.n+1))), axis = 0)
        nq = self.n + 1 # size of Fock space

        # CTQW dynamics is simulated by solving the GKSL eq, with Hamiltonian H = h_bar*A^q
        H = Aq
        H = qt.Qobj(H)
        psi0 = qt.basis(nq, 0) # change to args later on
        rho0 = psi0 * psi0.dag() # init cond rho0 = |1><1|, but with zero indexing
        L = qt.basis(nq, self.n) * qt.basis(nq, self.target).dag() # L = |n+1><t|, but with zero indexing
        c_op = np.sqrt(gamma) * L # collapse op C = sqrt(gamma)*L
        options = qt.Options(nsteps=1500, store_states=True, atol=1e-12, rtol=1e-12)
        result = qt.mesolve(H, rho0, self.t, c_op, progress_bar=True,options = options) # solves the master eq
        
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
            f = np.dot(temp2, p0)*temp1 #eq (3) Melnikov 1
            self.pc[i] = f[self.target,0]


def example_main():
    corpus = ExampleCorpus()
    corpus.generate_graphs()
    corpus.plot_corpus()



example_main()
        
        