# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 17:58:59 2020

@author: hanna
"""

import numpy as np
import qutip as qt
from scipy.linalg import expm
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from matplotlib import animation


class GraphSimulation():
    '''    
    Contains methods for simulating the quantum random walk and for classical random walk.
    Init will run simulations and categorise the graph.
    
    Args:
            initial : initial vertex for the start of random walk (integer) default 0
            target : target vertex for the finish of the quantum walk (integer) default 1
            graph : [A, node_list] A is a nxn numpy array of the adjecency matrix, node_list is the list of nodes
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
        self.max_time = step_size * 100 * self.n
        self.max_steps = 100 * self.n
        self.initial = initial
        self.target = target
        self.pth = 1/np.log(self.n)
        
        self.c_hitting_time, self.p_c = self.CRW_simulation() # run until finished or max_time
        
        self.q_hitting_time, self.p_q = self.QRW_simulation(ancilla = False) # vanilla
        
        if self.magic:
            self.positive_hitting_time, self.p_positive = self.QRW_simulation(ancilla = True, connection = 'bidirected', split = False, superposition = 'positive')
            self.negative_hitting_time, self.p_negative = self.QRW_simulation(ancilla = True, connection = 'bidirected', split = False, superposition = 'negative')
            self.T_hitting_time, self.p_T = self.QRW_simulation(ancilla = True, connection = 'bidirected', split = False, superposition = 'T')
            self.H_hitting_time, self.p_H = self.QRW_simulation(ancilla = True, connection = 'bidirected', split = False, superposition = 'H')
            self.names = ['c', 'q', 'positive', 'negative', 'T', 'H']
            self.colors = ['grey', 'magenta', 'cyan', 'yellow', 'red', 'blue']
        else:
            self.names = ['c', 'q']
            self.colors = ['grey', 'magenta']

        self.hitting_time_all = self.c_hitting_time
        self.p_all = [self.p_c]
        for name in self.names[1:]:
            self.hitting_time_all = np.append(self.hitting_time_all, eval('self.' + name + '_hitting_time'), axis = 1)
            self.p_all.append(eval('self.p_' + name))
        self.set_label()
        self.set_ranking()

    def set_label(self):
        temp = np.where(self.hitting_time_all[1, :], self.hitting_time_all[0, :], self.max_time)
        self.label = np.where(temp == temp.min(), 1.0, 0.0) # to get the ties

    def set_ranking(self):
        temp = np.where(self.hitting_time_all[1, :], self.hitting_time_all[0, :], self.max_time)
        self.ranking = rankdata(temp, method = 'dense')

        prob_at_end = np.array([x[-1, -1] for x in self.p_all])
        if self.magic:
            for i in range(2,6):
                prob_at_end[i] = self.p_all[i][-1, self.n] # sink instead of extra node

        not_found = np.where(np.where(self.hitting_time_all[1, :] == 0., True, False), prob_at_end, 0) # [0, 30, 0, 0, 20, 0]
        for l in range(np.sum(np.where(self.hitting_time_all[1, :] == 0., True, False))): # falses should be last in ranking
            idx = np.argmax(not_found)
            self.ranking[idx] = np.amax(self.ranking) if l == 0 else np.amax(self.ranking) + 1
            not_found[idx] = 0

    
    def CRW_simulation(self):
        c_hitting_time = 0
        p_c = np.zeros((1, self.n)) # classical probability, dim=(time, node)
        p_c[0, self.initial] = 1
        Ac = np.copy(self.A)
        Ac[:, self.target] = 0
        Ac[self.target, self.target] = 1
        T = np.copy(Ac) # transition matrix
        for col in range(self.n):
            T[:,col] = T[:,col] / np.sum(Ac[:,col]) if np.sum(Ac[:,col]) != 0 else np.zeros(self.n) # div by 0!!!
        p0 = np.zeros((self.n,1))
        p0[self.initial] = 1
        
        t = self.step_size # don't need to calculate t = 0 as that is p0
        prob = 0
        while (prob < self.pth) and (t < self.max_time): # gives RuntimeWarning: underflow at t = 709, it is ok, we still want QRW-sim to run
            temp1 = np.exp(-t)
            temp2 = expm(T*t)
            temp3 = np.dot(temp2, p0)*temp1 #eq (3) Melnikov 1
            p_c = np.append(p_c, np.reshape(temp3, (1, self.n)), axis = 0)
            prob = temp3[self.target]
            t = round(t + self.step_size, 6)
        c_hitting_time = t
        self.t_steps = int(np.ceil(c_hitting_time / self.step_size))
        found = p_c[self.t_steps-1, self.target] > self.pth
        return np.array([[c_hitting_time], [found]]), p_c


    def QRW_simulation(self, gamma = 1, ancilla = False, connection = 'bidirected', split = False, superposition = 'no superposition', verbose = False):
        '''
        Args:
                connection : 'ghost' (not connected ancillary node, only with split), or
                             'bidirected' (connected undirected ancillary node).
                split : True or False (the ancillary node is connected to the initial nodes neighbours).
                superposition :
                              'positive' |+> = sqrt(1/2) * (|i> + |e>)
                              'negative' |-> = sqrt(1/2) * (|i> - |e>)
                              'T' where T_state = cos(beta) * |i> + exp(1j*(pi/4)) * sin(beta) * |e>, beta = 0.5 * arccos(1/sqrt(3))), or
                              'H' where H_state = cos(np.pi/8) * |i> + sin(np.pi/8) * |e>).
        '''        
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
        options = qt.Options(store_states=True, atol=1e-12, rtol=1e-12)
        

        result = qt.mesolve(H, rho0, np.linspace(0, self.max_time, self.max_steps), c_op, options = options) # solves the master eq

        p_q = [s.diag().real for s in result.states]
        p_q = np.array(p_q)
        q_hitting_step = np.argmax(p_q[:, self.sink] > self.pth)
        self.t_steps = q_hitting_step if self.t_steps < q_hitting_step else self.t_steps
        q_hitting_time = round(q_hitting_step*self.step_size, 6) if p_q[q_hitting_step, self.sink] != 0 else np.argmax(p_q[:, self.sink])
        found = p_q[q_hitting_step, self.sink] > self.pth

        return np.array([[q_hitting_time], [found]]), p_q
                
    
    def plot_p(self):
        for i in range(1, len(self.p_all)):
            p = self.p_all[i]
            plt.plot(np.linspace(0, p.shape[0], p.shape[0]), p[:, self.sink], label = self.names[i], c = self.colors[i])
        p = self.p_c[:, self.target]
        plt.plot(np.linspace(0, p.shape[0], p.shape[0]), p, '-.', label = self.names[0], c = self.colors[0])
        plt.title('Probability in the target(sink), label:' + str(self.label) + ', ranking:' +str(self.ranking))
        plt.xlabel('time')
        plt.ylabel('probability')
        plt.plot(np.linspace(0, self.max_steps,self.max_steps), np.ones(self.max_steps)*self.pth, '--', color = (0, 0, 0), label = 'pth')
        plt.legend()
        plt.savefig('probability_n'+str(self.n)+'_R' + str(np.random.randint(0,100)) )

    def animate_p(self, p_list = [0, 1]):
        '''
        Animate the probabilities given in p_list over the nodes.
        [0, 1] are p_c and p_q.
        ['c', 'q', 'pos', 'neg', 'T', 'H'] use the indices for the rest.
        p_list = 'all' gives all probabilities.
        '''
        if p_list == 'all':
            p_list = [0, 1, 2, 3, 4, 5] 

        # set up the figure, the axis, and the plot element we want to animate
        fig = plt.figure()
        ax = plt.axes(xlim=(0, self.n-1), ylim=(-0.1, 1.0))
        pthline = ax.plot([], [], lw=2, label= 'threshold', c = 'black')[0]
        if p_list == [0]:
            target = ax.plot([], [], mew=7, label = 'target', c = 'r', marker = 'o')[0]
        else:
            target = ax.plot([], [], mew=7, label = 'target/sink', c = 'r', marker = 'o')[0]
        initial = ax.plot([], [], mew=7, label = 'initial', c = 'y', marker = 'v')[0]
        line_list = []
        for p in p_list:
            line_list.append(ax.plot([], [], lw=2, label= self.names[p], c = self.colors[p])[0])
        ax.set_xlabel('nodes')
        ax.set_ylabel('probability')
        ax.set_yticks([0, 0.5, 1.0])
        plt.title('Probability over a linear graph with label ' + str(self.label))
        plt.legend()

                                                                                                                        
        # initialization function: plot the background of each frame
        def init():
            p_return = []
            for p in line_list:
                p_return.append(p.set_data([], []))
            pthline.set_data([], [])
            target.set_data([], [])
            initial.set_data([], [])
            return target, initial, pthline, p_return

        # animation function called sequentially
        def animate(t):
            p_return = []
            for p in enumerate(p_list):
                if p[1] == 0:
                    p_c_anim = self.p_c[t,:][np.array(self.node_list)]
                    line_list[0].set_data(np.arange(0, self.n), p_c_anim)
                else: # sink instead of target
                    p_q = eval('self.p_' + self.names[p[0]])
                    anim = np.copy(p_q[:, :self.n]) 
                    anim[:, self.target] = p_q[:, self.sink] 
                    anim = anim[t, :][np.array(self.node_list)] 
            
                    line_list[p[0]].set_data(np.arange(0, self.n), anim)
                p_return.append(line_list[p[0]])


            pthline.set_data(np.arange(0, self.n), self.pth)
            target.set_data(np.where(np.array(self.node_list) == self.target), 0)
            initial.set_data(np.where(np.array(self.node_list) == self.initial), 0)

            return target, initial, pthline, p_return

        # call the animator.  blit=True means only re-draw the parts that have changed.
        frames = self.t_steps
        for i in p_list:
            frames = self.p_all[i].shape[0] if self.p_all[i].shape[0] < frames else frames


        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=frames, interval=1, repeat=True)

        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=40, metadata=dict(artist='Me'), bitrate=1800)
        anim.save('probability_wave_n'+ str(self.n) + '_R' + str(np.random.randint(0,100)) + '.mp4', writer=writer) # give a random number to save all movies
        #plt.savefig('probability_wave_n'+ str(self.n) + '_R' + str(np.random.randint(0,100)) + "plot.png")

        '''
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=20, interval=1, repeat=True)
        anim.save('probability_wave_n'+ str(self.n) + '_R' + str(np.random.randint(0,100)) + '.mp4', writer=writer) # give a random number to save all movies
        plt.savefig('probability_wave_n'+ str(self.n) + '_R' + str(np.random.randint(0,100)) + "plot.png")
        '''

