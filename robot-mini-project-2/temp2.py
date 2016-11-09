# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 18:29:13 2016

@author: sleeba
"""
import numpy as np
import robot
#transition_matrix=[[.25,.75,0],[0,.25,.75],[0,0,1]]
#observation_matrix=[[1,0],[0,1],[1,0]]
#prior_forward={1:1/3,2:1/3,3:1/3}
#prior_backward={1:1/3,2:1/3,3:1/3}
#observations=[(0),(1),(0)]
#all_possible_hidden_states=[1,2,3]
#all_possible_observed_states=[0,1]

all_possible_hidden_states = robot.get_all_hidden_states()
all_possible_observed_states = robot.get_all_observed_states()
prior_distribution = robot.initial_distribution()
transition_model = robot.transition_model
observation_model = robot.observation_model
a=len(all_possible_hidden_states)
b=len(all_possible_observed_states)
#observation matrix formation
observation_matrix= np.zeros(shape=(a,b))
count=0
for i in all_possible_hidden_states:
    out=observation_model(i)
    for j in out:
        x,y=j
        idx=all_possible_observed_states.index((x,y))
        observation_matrix[count][idx]=out[j]
    count+=1
    
#transition matrix formation
    
transition_matrix= np.zeros(shape=(a,a))
count=0  
for i in all_possible_hidden_states:
    out=transition_model(i)
    for j in out:
        idx=all_possible_hidden_states.index(j)
        transition_matrix[count][idx]=out[j]
    count+=1

#transition matrix for backward messages
    
back_transition_matrix=np.transpose(transition_matrix)

#prior for backward messages

prior_backward=robot.Distribution()
for i in all_possible_hidden_states:
    prior_backward[i]=1/440

# Functions for you to implement
#

def forward_backward(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    robot.py and see how it is used in both robot.py and the function
    generate_data() above, and the i-th Distribution should correspond to time
    step i
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #
    num_time_steps=len(observations)
    forward_messages = [None] *len(all_possible_hidden_states)
    forward_messages[0] = [prior_distribution]
    # TODO: Compute the forward messages
    for i in range(1,num_time_steps):
        iteration_sum=[0] * len(all_possible_hidden_states)
        intermediate = [None] * len(all_possible_hidden_states)
        final_sum=[0] * len(all_possible_hidden_states)
        x,y=observations[i-1]
        message=robot.Distribution()
        index=all_possible_observed_states.index((x,y))
        for j in range(len(all_possible_hidden_states)):
            intermediate[j]=forward_messages[i-1][0][all_possible_hidden_states[j]]*observation_matrix[j][index]
        for k in range(len(all_possible_hidden_states)):
            for l in range(len(all_possible_hidden_states)):
                iteration_sum[l]=intermediate[k]*transition_matrix[k][l]
            for m in range(len(all_possible_hidden_states)):
                final_sum[m]=final_sum[m]+iteration_sum[m]       
        pranthu=0
        for j in all_possible_hidden_states:
            message[j]=final_sum[pranthu]
            pranthu+=1
        print(message)
        print('\n\n\n')
        forward_messages[i]=[message]
    
    backward_messages = [None] * num_time_steps
    backward_messages[num_time_steps-1]=[prior_backward]
    # TODO: Compute the backward messages\
    for i in range(num_time_steps-1,0,-1):
        iteration_sum=[0] * len(all_possible_hidden_states)
        intermediate = [None] * len(all_possible_hidden_states)
        final_sum=[0] * len(all_possible_hidden_states)
        x,y=observations[i]
        message=robot.Distribution()
        index=all_possible_observed_states.index((x,y))
        for j in range(len(all_possible_hidden_states)):
            intermediate[j]=backward_messages[i][0][all_possible_hidden_states[j]]*observation_matrix[j][index]
        for k in range(len(all_possible_hidden_states)):
            for l in range(len(all_possible_hidden_states)):
                iteration_sum[l]=intermediate[k]*back_transition_matrix[k][l]
            for m in range(len(all_possible_hidden_states)):
                final_sum[m]=final_sum[m]+iteration_sum[m]
        pranthu=0
        for j in all_possible_hidden_states:
            message[j]=final_sum[pranthu]
            pranthu+=1
        backward_messages[i-1]=[message]

    marginals = [None] * num_time_steps # remove this
    # TODO: Compute the marginals 
    marginal_distribution=robot.Distribution()
    
    for i in range(num_time_steps):
        x,y=observations[i]
        index=all_possible_observed_states.index((x,y))
        if i>0 and i<num_time_steps-1:
            for j in range(len(all_possible_hidden_states)):
                marginal_distribution[all_possible_hidden_states[j]]=observation_matrix[j][index]*backward_messages[i][0][all_possible_hidden_states[j]]*forward_messages[i][0][all_possible_hidden_states[j]]
        if i==0:
            for j in range(len(all_possible_hidden_states)):
                marginal_distribution[all_possible_hidden_states[j]]=observation_matrix[j][index]*backward_messages[0][0][all_possible_hidden_states[j]]
        elif i==num_time_steps-1:
            for j in range(len(all_possible_hidden_states)):
                marginal_distribution[all_possible_hidden_states[j]]=observation_matrix[j][index]*forward_messages[num_time_steps-1][0][all_possible_hidden_states[j]]   
        marginals[i]=marginal_distribution
        marginals[i].renormalize()
    return marginals
    
print('Running forward-backward...')
marginals = forward_backward([(8,3),(8,4),(8,6)])
print("\n")

timestep = 0
print("Most likely parts of marginal at time %d:" % (timestep))
if marginals[timestep] is not None:
    print(sorted(marginals[timestep].items(),key=lambda x: x[1],reverse=True)[:10])
else:
    print('*No marginal computed*')
print("\n")
