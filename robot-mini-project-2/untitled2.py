# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 22:44:24 2016

@author: sleeba
"""

import numpy as np
import scipy.io
import robot
transition_matrix= np.zeros(shape=(a,a))
count=0  
for i in all_possible_hidden_states:
    out=transition_model(i)
    for j in out:
        idx=all_possible_hidden_states.index(j)
        transition_matrix[count][idx]=out[j]
    count+=1
observation_matrix= np.zeros(shape=(a,b))
count=0
for i in all_possible_hidden_states:
    out=observation_model(i)
    for j in out:
        x,y=j
        idx=all_possible_observed_states.index((x,y))
        observation_matrix[count][idx]=out[j]
    count+=1
index=all_possible_observed_states.index((8,3))
prior_matrix= np.zeros(shape=(a,1))
for i in all_possible_hidden_states:
    x,y,state=i
    if state=='stay':
        index=all_possible_hidden_states.index(i)
        prior_matrix[index][0]=1/96
        
iteration_sum=[0] * len(all_possible_hidden_states)
intermediate = [None] * len(all_possible_hidden_states)
obervations=(8,3)
final_sum=[0] * len(all_possible_hidden_states)
message=robot.Distribution()
num_time_steps=1
forward_messages = [None] * num_time_steps
forward_messages[0] = [prior_matrix]
if observations==None:
    index=0
    for j in range(len(all_possible_hidden_states)):
        intermediate[j]=forward_messages[i-1][0][all_possible_hidden_states[j]]*none_observation_matrix[j][index]
    for k in range(len(all_possible_hidden_states)):
        for l in range(len(all_possible_hidden_states)):
            iteration_sum[l]=intermediate[k]*transition_matrix[k][l]
        for m in range(len(all_possible_hidden_states)):
            final_sum[m]=final_sum[m]+iteration_sum[m]
else:
    x,y=(8,3)
    index=all_possible_observed_states.index((x,y))
    for j in range(len(all_possible_hidden_states)):
        intermediate[j]=forward_messages[0][0][all_possible_hidden_states[j]]*observation_matrix[j][index]
    for k in range(len(all_possible_hidden_states)):
        for l in range(len(all_possible_hidden_states)):
            iteration_sum[l]=intermediate[k]*transition_matrix[k][l]
        for m in range(len(all_possible_hidden_states)):
            final_sum[m]=final_sum[m]+iteration_sum[m]       
    pranthu=0
    for j in all_possible_hidden_states:
        message[j]=final_sum[pranthu]
        pranthu+=1
    out=list(message.values())
compare=np.zeros(shape=(a,1))
for i in range(len(all_possible_hidden_states)):
    compare[i][0]=out[i]
scipy.io.savemat('C:/Users/sleeba/Documents/MATLAB/MIT_assignment/compare.mat', mdict={'arr':compare})
#scipy.io.savemat('C:/Users/sleeba/Documents/MATLAB/observation.mat', mdict={'arr': observation_matrix})