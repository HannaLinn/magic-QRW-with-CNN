import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp
from scipy import constants
import numpy as np
import qutip as qt
from scipy.linalg import expm

# make a simple line graph, three nodes, from ex in OG1
# i is initial vertex for the walk and 2 is the target vertex
initial = 0
target = 2


# graph 1
G1 = nx.Graph()
nodes = [0, 1, 2]
G1.add_nodes_from(nodes, sink = False)
edges = [(0,2), (2,1)]
G1.add_edges_from(edges)

# graph 2
G2 = nx.Graph()
nodes = [0, 1, 2]
G2.add_nodes_from(nodes, sink = False)
edges = [(0,1), (1,2)]
G2.add_edges_from(edges)

# graph 3
G3 = nx.Graph()
nodes = [0, 1, 2]
G3.add_nodes_from(nodes, sink = False)
edges = [(2,0), (0,1)]
G3.add_edges_from(edges)

# ---------------------------
G_list = [G1, G2, G3]

for g in range(3):
    G = G_list[g]
    A = nx.to_numpy_matrix(G)
    n = len(G)
    #nx.draw_networkx(G, with_labels=True)
    
    # time
    t = np.linspace(0.0, 10.0, 100)
    t_steps = t.size
    
    # Classical part
    Ac = np.copy(A)
    Ac[:,target] = 0
    Ac[target,target] = 1
    T = np.copy(Ac)
    for col in range(n):
        T[:,col] = T[:,col] / np.sum(Ac[:,col])
    
    p0 = np.zeros((n,1))
    p0[initial] = 1
    pc = np.zeros((n, t_steps))
    for i in range(t_steps):
        temp1 = np.exp(-t[i])
        temp2 = expm(T*t[i])
        temp3 = np.dot(temp2, p0)
        f = temp3*temp1 #eq (3)
        pc[:,i] = f[:,0].T
    
    
    plt.figure(0)
    plt.title('Classical particle')
    plt.ylabel('detection probabibilies, pc')
    plt.xlabel('time, t')
    plt.plot(t, pc[target,:], label = 'graph ' + str(g+1))
    plt.legend()


    # Quantum part
    
    Gq = G.copy()
    # add sink node
    Aq = nx.to_numpy_matrix(Gq)
    Aq = np.concatenate((Aq, np.zeros((n,1))), axis = 1)
    Aq = np.concatenate((Aq, np.zeros((1, n+1))), axis = 0)
    nq = n + 1 # size of Fock space
    
    # do qutip
    # CTQW dynamics is simulated by solving the GKSL eq, with Hamiltonian H = h_bar*A^q
    H = Aq # not Qobj
    print(constants.hbar)
    H = qt.Qobj(H)
    #qt.hinton(H)
    gamma = 1
    psi0 = qt.basis(nq, 0)
    rho0 = psi0 * psi0.dag() # init cond rho0 = |1><1|, but with zero indexing
    L = qt.basis(nq, n) * qt.basis(nq, n-1).dag() # L = |n+1><n|, but with zero indexing
    c_op = np.sqrt(gamma) * L # collapse op C = sqrt(gamma)*L
    result = qt.mesolve(H, rho0, t, c_op, progress_bar=True,options = qt.Options(nsteps=1500, store_states=True, atol=1e-12, rtol=1e-12)) # solves the master eq
    pq = np.zeros((t_steps, 1))
    for i in range(t_steps):
        temp = result.states[i].full()
        pq[i] = temp[n,n] #pq = rho[n+1][n+1], [n,n] in zero indexing
        
    plt.figure(1)
    plt.title('Quantum particle')
    plt.ylabel('detection probabibilies, pq')
    plt.xlabel('time, t')
    plt.plot(t, pq, label = 'graph ' + str(g+1))
    plt.legend()

# Compare pq and pc to pth, page 8 og1

pth = 1/sp.log(n)

