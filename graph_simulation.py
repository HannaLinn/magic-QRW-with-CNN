# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 17:58:59 2020

@author: hanna
"""

import numpy as np
import qutip as qt
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib import animation


class GraphSimulation():
    '''    
    Contains methods for simulating the quantum random walk and for classical random walk.
    Init will run simulations and categorise the graph.
    
    Args:
            initial : initial vertex for the start of random walk (integer) default 0
            target : target vertex for the finish of the quantum walk (integer) default 1
            A : is a nxn numpy array of the adjecency matrix
            label : numpy array of [classical_advantage, quantum_advantage] as a float
                    [1.0, 0.0] is classical
                    [0.0, 1.0] is quantum
                    [0.0, 0.0] is a tie
            
    '''
    def __init__(self, graph, step_size = 0.10, initial = 0, target = 1, magic = False):
        self.A = graph[0]
        self.n = self.A.shape[0]
        self.node_list = graph[1]
        self.magic = magic
        self.step_size = step_size
        #self.max_time = step_size * 1000 * self.n
        self.max_time = 1000 * self.n
        self.initial = initial
        self.target = target
        self.pth = 1/np.log(self.n)
        
        self.pc_hitting_time, self.pc = self.CRW_simulation() # run until finished or max_time
        
        self.pq_q_hitting_time, self.pq_q = self.QRW_simulation(ancilla = False) # vanilla

        if self.magic:
            self.pq_pos_hitting_time, self.pq_pos = self.QRW_simulation(ancilla = True, connection = 'bidirected', split = False, superposition = 'positive')
            self.pq_neg_hitting_time, self.pq_neg = self.QRW_simulation(ancilla = True, connection = 'bidirected', split = False, superposition = 'negative')
            self.pq_T_hitting_time, self.pq_T = self.QRW_simulation(ancilla = True, connection = 'bidirected', split = False, superposition = 'T')
            self.pq_H_hitting_time, self.pq_H = self.QRW_simulation(ancilla = True, connection = 'bidirected', split = False, superposition = 'H')
            self.names = ['c', 'q', 'pos', 'neg', 'T', 'H']
            self.colors = ['grey', 'magenta', 'cyan', 'yellow', 'red', 'blue']
        else:
            self.names = ['c', 'q']
            self.colors = ['grey', 'magenta']

        self.hitting_time_all = np.array([self.pc_hitting_time])
        self.p_all = np.array([self.pc])
        for name in self.names[1:]:
            self.hitting_time_all = np.append(self.hitting_time_all, eval('self.pq_' + name + '_hitting_time'))
            self.p_all = np.append(self.p_all, eval('self.pq_' + name))
        #self.hitting_time_all = np.array([self.pc_hitting_time, self.pq_hitting_time, self.pq_pos_hitting_time, self.pq_neg_hitting_time, self.pq_T_hitting_time, self.pq_H_hitting_time])
        #self.p_all = [self.pc, self.pq, self.pq_pos, self.pq_neg, self.pq_T, self.pq_H]
        self.set_label()

    def set_label(self):
        self.label = np.where(self.hitting_time_all == self.hitting_time_all.min(), 1.0, 0.0)

    
    def CRW_simulation(self):
        pc_hitting_time = 0
        pc = np.zeros((1, self.n)) # classical probability, dim=(time, node)
        pc[0, self.initial] = 1
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
        while (prob < self.pth) and (t < self.max_time): # gives RuntimeWarning: underflow at t = 709, it is ok, we still want QRW-sim to run
            temp1 = np.exp(-t)
            temp2 = expm(T*t)
            temp3 = np.dot(temp2, p0)*temp1 #eq (3) Melnikov 1
            pc = np.append(pc, np.reshape(temp3, (1, self.n)), axis = 0)
            prob = temp3[self.target]
            t = round(t + self.step_size, 6)
        pc_hitting_time = t
        self.t_steps = int(np.ceil(pc_hitting_time / self.step_size))
        self.t = np.linspace(0.0, pc_hitting_time, self.t_steps)
        return pc_hitting_time, pc


    def QRW_simulation(self, gamma = 1, ancilla = False, connection = 'bidirected', split = False, superposition = 'positive', verbose = False):
        '''
        Args:
                connection : 'ghost' (not connected ancillary node, only with split), or
                             'bidirected' (connected undirected ancillary node).
                split : True or False (the ancillary node is connected to the initial nodes neighbours).
                superposition :
                              'positive' (np.sqrt(1/2) * (qt.basis(nq, self.initial) + qt.basis(nq, self.nq-1)))
                              'negative' (np.sqrt(1/2) * (qt.basis(nq, self.initial) - qt.basis(nq, self.nq-1)))
                              'T' (where T_state = np.cos(beta) * qt.basis(nq, self.initial) + np.exp(1j*(np.pi/4)) * np.sin(beta) * qt.basis(nq, nq-1), beta = 0.5 * np.arccos(1/np.sqrt(3))), or
                              'H' (where H_state = np.cos(np.pi/8) * qt.basis(nq, self.initial) + np.sin(np.pi/8) * qt.basis(nq, nq-1)).
        '''
        pq_hitting_time = self.step_size * 1000 * self.n
        pq = np.zeros((self.t_steps, self.n+1)) # quantum probability, dim=(time, node+1), +1 for sink
        Aq = np.copy(self.A)
        Aq = np.concatenate((Aq, np.zeros((self.n, 1))), axis = 1) # adding sink
        Aq = np.concatenate((Aq, np.zeros((1, self.n+1))), axis = 0) # adding sink
        nq_dim = self.n + 1 # dimension of Fock space
        self.sink = self.n # index of the sink with 0 indexing

        if ancilla:

            Aq = np.concatenate((Aq, np.zeros((self.n+1,1))), axis = 1) # ancilla node
            Aq = np.concatenate((Aq, np.zeros((1, self.n+2))), axis = 0) # ancilla node
            nq_dim = self.n + 2 # size of Fock space
            self.extra = self.n + 1
            if connection == 'bidirected':
                Aq[0, self.extra] = 1
                Aq[self.extra, 0] = 1

            if split:
                for row in range(self.n-1):
                    if self.A[row, self.initial]:
                        Aq[self.extra, row] = 1
                        Aq[row, self.extra] = 1
            else:
                pass

            if superposition == 'positive':
                rho = np.sqrt(1/2) * (qt.basis(nq_dim, self.initial) + qt.basis(nq_dim, self.extra))
            elif superposition == 'negative':
                rho = np.sqrt(1/2) * (qt.basis(nq_dim, self.initial) - qt.basis(nq_dim, self.extra))
            elif superposition == 'T':
                beta = 0.5 * np.arccos(1/np.sqrt(3))
                rho = np.cos(beta) * qt.basis(nq_dim, self.initial) + np.exp(1j*(np.pi/4)) * np.sin(beta) * qt.basis(nq_dim, self.extra)
            elif superposition == 'H':
                rho = np.cos(np.pi/8) * qt.basis(nq_dim, self.initial) + np.sin(np.pi/8) * qt.basis(nq_dim, self.extra)
            else:
                raise NameError('Name the type of magic state: positive, negative, T, or H.')
            rho0 = rho * rho.dag()

            if verbose:
                #print('Aq: ', Aq)
                qt.hinton(rho0)
                plt.title(superposition)
                plt.show()
        else:
            psi0 = qt.basis(nq_dim, self.initial)
            rho0 = psi0 * psi0.dag() # init cond rho0 = |1><1|, but with zero indexing

        # CTQW dynamics is simulated by solving the GKSL eq, with Hamiltonian H = h_bar*A^q, h_bar=1
        H = qt.Qobj(Aq)
        L = qt.basis(nq_dim, self.sink) * qt.basis(nq_dim, self.target).dag() # L = |n+1><t|, but with zero indexing
        c_op = np.sqrt(gamma) * L # collapse op C = sqrt(gamma)*L
        options = qt.Options(nsteps=1500, store_states=True, atol=1e-12, rtol=1e-12)
        result = qt.mesolve(H, rho0, self.t, c_op, options = options) # solves the master eq
        
        found = False
        for t in range(self.t_steps):
            pq[t, :] = result.states[t].full().diagonal()[:self.sink+1].real # diagonal is the probability for the different nodes
            prob = result.states[t].full().diagonal()[self.sink].real
            if (prob > self.pth) and (not found):
                pq_hitting_time = round(t * self.step_size, 6)
                found = True

        return pq_hitting_time, pq
                
    
    def plot_p(self):
        plt.plot(self.t, self.pc[:, self.target], '--', label = self.names[0], c = self.colors[0])
        for i in range(1, len(self.p_all)-1):
            p = self.p_all[i]
            plt.plot(self.t, p[:, -1], label = self.names[i], c = self.colors[i])
        plt.title('Probability in the target (sink) node, labeled ' + str(self.label))
        plt.xlabel('time')
        plt.ylabel('probability')
        plt.plot(self.t, np.ones(self.t_steps)*self.pth, '.', color = (0, 0, 0), label = 'pth')
        plt.legend()

    def animate_p(self):
        # Animate the probability over the nodes.

        # set up the figure, the axis, and the plot element we want to animate
        fig = plt.figure()
        ax = plt.axes(xlim=(0, self.n-1), ylim=(-0.1, 1.0))
        pthline, = ax.plot([], [], lw=2, label= 'threshold', c = 'black')
        pcline, = ax.plot([], [], lw=2, label= 'classical', c = 'grey')
        pqline, = ax.plot([], [], lw=2, label = 'quantum', c = 'm')
        target, = ax.plot([], [], mew=7, label = 'target', c = 'r', marker = 'o')
        initial, = ax.plot([], [], mew=7, label = 'initial', c = 'y', marker = 'v')
        ax.set_xlabel('nodes')
        ax.set_ylabel('probability')
        plt.title('Probability over a line graph')
        plt.legend()

                                                                                                                        
        # initialization function: plot the background of each frame
        def init():
            pcline.set_data([], [])
            pqline.set_data([], [])
            pthline.set_data([], [])
            target.set_data([], [])
            initial.set_data([], [])
            return pcline, pqline, target, initial, pthline

        # animation function called sequentially
        def animate(t):
            pc_anim = self.pc[t,:][np.array(self.node_list)]
            pq_anim = np.copy(self.pq_q[:, :self.n]) 
            pq_anim[:, self.target] = self.pq_q[:, self.sink] # sink instead of target
            pq_anim = pq_anim[t, :][np.array(self.node_list)] 
            pth_anim = self.pth
            pcline.set_data(np.arange(0, self.n), pc_anim)
            pqline.set_data(np.arange(0, self.n), pq_anim)
            pthline.set_data(np.arange(0, self.n), pth_anim)

            target.set_data(np.where(np.array(self.node_list) == self.target), 0)
            initial.set_data(np.where(np.array(self.node_list) == self.initial), 0)

            return pcline, pqline, target, initial,


        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=self.t_steps, interval=1, repeat=True)

        #plt.show()
        #plt.savefig("./plot.png")             # store final frame
        #anim.save('wavepacketanimation'+ str(self.n) + '.gif', writer='imagemagick', fps=30)
        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
        anim.save('wavepacketanimation'+ str(self.n) + '.mp4', writer=writer)

