# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:46:09 2020

@author: hanna

Timing the time it takes to generate corpuses.
"""

import corpus_v4 as c4
import corpus_v5 as c5
import time

import matplotlib.pyplot as plt
import numpy as np


'''
n = 25
N = 100
number = 5

time4_sum = 0
time5_sum = 0

for i in range(number):
    corpus4 = c4.Corpus_n(n = n)
    start = time.time()
    corpus4.generate_random_graphs(N = N, total_random=True)
    end = time.time()
    time4 = end-start
    time4_sum += time4
    
    corpus5 = c5.Corpus_n(n = n)
    start = time.time()
    corpus5.generate_random_graphs(N = N, total_random=True)
    end = time.time()
    time5 = end-start
    time5_sum += time5
    
    print('progress ', i)

print('time4 ', time4_sum/number)
print('time5 ', time5_sum/number)
'''

'''
y axis is time it took
x axis is number of nodes
'''

max_nodes = 25
N = 100
number = 10


times = np.zeros((max_nodes, 1))
percentage_q = np.zeros((max_nodes, 1))

for n in range(3, max_nodes):
    time5_sum = 0
    q_counter_sum = 0
    for i in range(number):
        corpus5 = c5.Corpus_n(n = n)
        start = time.time()
        corpus5.generate_random_graphs(N = N, total_random=True)
        end = time.time()
        time5 = end-start
        time5_sum += time5
        
        q_counter = 0
        for g in corpus5.corpus_list:
            if g.label[1] > 0:
                q_counter +=1
        q_counter_sum += q_counter/N
        
    print('-'*10, 'progress in nodes', n, '-'*10)
    times[n] = time5_sum/number
    percentage_q[n] = q_counter_sum/number
    
plt.figure(0)
plt.xlabel('number of nodes')
plt.ylabel('time')
plt.plot(times)

plt.figure(1)
plt.xlabel('number of nodes')
plt.ylabel('percentage of q')
plt.plot(percentage_q)