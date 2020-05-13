# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:55:14 2020

@author: hanna
"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy.linalg import expm


class Corpus(object):
    '''
    Builds the corpus with the help of the Graph class.
    '''
    def __init__(self, initial = 0, target = 1):
        self.corpus_list = []
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
    def gen_line_graph_lists(self, n):
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
                    rest = [x for x in range(0,n) if x != self.target and x != self.initial]
                    for k in range(0,n):
                        if k != t and k != i:
                            node_list[k] = rest[l]
                            l += 1
                    # add and restart
                    grande_list.append(node_list)
                    node_list = ['0'] * n
                    node_list[i] = self.initial
        return grande_list
    
    def plot_line_graphs(self, uncategorized_G):
        # Plot the graphs
        N = len(uncategorized_G)
        fig, axes = plt.subplots(nrows=1, ncols=N)
        ax = axes.flatten()
        for i in range(N):
            nx.draw_networkx(uncategorized_G[i], ax=ax[i], with_labels=True)
            ax[i].set_axis_off()
            
        plt.figure(N+2)
        for i in range(N):
            plt.plot(self.corpus_list[i].t, self.corpus_list[i].pq, '-', color = (i/N, 0.2, 0.2), label = str(i) +' pq, quantum')

        for i in range(N):
            plt.plot(self.corpus_list[i].t, self.corpus_list[i].pc, '--', color = (i/N, 0.2, 0.2), label = str(i) + ' pc, classical')
        plt.legend()

    '''
    1. generate graphs
    2. categorise
    3. plot
    '''

    def generate_graphs(self, n):
        uncategorized_G = []
        
        # Line graphs
        graph_list = self.gen_line_graph_lists(n)
        for g in graph_list:
            G = nx.Graph()
            nodes = range(n)
            G.add_nodes_from(nodes)
            for i in range(n-1):
                G.add_edge(g[i], g[i+1])

            uncategorized_G.append(G)
            

        nx.draw_networkx(uncategorized_G[0], with_labels=True)
        
        self.categorise(uncategorized_G)
        
        self.plot_line_graphs(uncategorized_G)
        
    '''
    Calls the simulation class, GraphSimulation and adds the categorised graohs to the list.
    '''
    def categorise(self, uncategorized_G):
        progress = 0
        for g in uncategorized_G:
            gcat = GraphSimulation(g)
            gcat.QRW_simulation()
            gcat.CRW_simulation()
            self.corpus_list.append(gcat)
            
            progress += 1
            print('Progress: ', progress/len(uncategorized_G))

class GraphSimulation(object):
    '''    
    Contains methods for simulating the quantum random walk and for classical random walk.
    
    Args:
            graph : networkx object
            time : stop (float) default 10.0, steps (integer) default 100
            initial : initial vertex for the start of random walk (integer) default 0
            target : target vertex for the finish of the quantum walk (integer) default 1
            
    How to comment? Comment more, look at comments of classes on the internet.
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
            f = np.dot(temp2, p0)*temp1 #eq (3) Melnikov 1
            self.pc[i] = f[self.target,0]

def main():
    corpus = Corpus(target = 1, initial = 0)
    corpus.generate_graphs(n=5)

main()