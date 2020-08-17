# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 17:58:59 2020

@author: hanna
"""

import numpy as np
import qutip as qt
from scipy.linalg import expm
import networkx as nx

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
                    [1.0, 0.0] is classical
                    [0.0, 1.0] is quantum
                    [0.0, 0.0] is a tie
            
    '''
    def __init__(self, G, step_size = 0.10, initial = 0, target = 1):
        self.G = G
        self.A = nx.to_numpy_matrix(G)
        self.n = len(G)
        self.step_size = step_size
        #self.max_time = step_size * 1000 * self.n
        self.max_time = 10000 * self.n
        self.initial = initial
        self.target = target
        self.pth = 1/np.log(self.n)
        
        self.pc_hitting_time = 0
        self.pc = np.zeros((1,1)) # classical probability at target after simulation
        self.CRW_simulation() # run until finished or max_time
        
        self.pq_hitting_time = step_size * 1000 * self.n
        self.t_steps = int(np.ceil(self.pc_hitting_time / self.step_size))
        self.t = np.linspace(0.0, self.pc_hitting_time, self.t_steps)
        self.pq = np.zeros((self.t_steps, 1)) # quantum probability at target after simulation
        self.QRW_simulation()
        

        if self.pc_hitting_time < self.pq_hitting_time:
            self.label = np.array([1.0, 0.0]) # classical better
        elif self.pc_hitting_time > self.pq_hitting_time:
            self.label = np.array([0.0, 1.0]) # quantum better
        elif self.pc_hitting_time >= self.max_time: #tie
            self.label = np.array([0.0, 0.0])
        else: #tie
            self.label = np.array([0.0, 0.0])

        #print('self.pc_hitting_time, self.pq_hitting_time ', self.pc_hitting_time, ' ', self.pq_hitting_time)
        
        
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
            if (self.pq[i] > self.pth) and not found:
                self.pq_hitting_time = float(self.pq[i])
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
