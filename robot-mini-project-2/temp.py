#!/usr/bin/env python
# inference.py
# Base code by George H. Chen (georgehc@mit.edu) -- updated 10/18/2016
import collections
import sys

import graphics
import numpy as np
import robot


# Throughout the code, we use these variables.
# Do NOT change these (but you'll need to use them!):
# - all_possible_hidden_states: a list of possible hidden states
# - all_possible_observed_states: a list of possible observed states
# - prior_distribution: a distribution over states
# - transition_model: a function that takes a hidden state and returns a
#     Distribution for the next state
# - observation_model: a function that takes a hidden state and returns a
#     Distribution for the observation from that hidden state
all_possible_hidden_states = robot.get_all_hidden_states()
all_possible_observed_states = robot.get_all_observed_states()
prior_distribution = robot.initial_distribution()
transition_model = robot.transition_model
observation_model = robot.observation_model

#observation matrix formation

a=len(all_possible_hidden_states)
b=len(all_possible_observed_states)
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
    prior_backward[i]=1/len(all_possible_hidden_states)

    
# You may find this function helpful for computing logs without yielding a
# NumPy warning when taking the log of 0.
def careful_log(x):
    # computes the log of a non-negative real number
    if x == 0:
        return -np.inf
    else:
        return np.log(x)


# -----------------------------------------------------------------------------
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

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    forward_messages[0] = [prior_distribution]

    #if None is the observation
    none_observation_matrix= np.ones(shape=(a,1))
    # TODO: Compute the forward messages
    for i in range(1,num_time_steps):
        iteration_sum=[0] * len(all_possible_hidden_states)
        intermediate = [None] * len(all_possible_hidden_states)
        final_sum=[0] * len(all_possible_hidden_states)
        message=robot.Distribution()
        if observations[i-1]==None:
            index=0
            for j in range(len(all_possible_hidden_states)):
                intermediate[j]=forward_messages[i-1][0][all_possible_hidden_states[j]]*none_observation_matrix[j][index]
            for k in range(len(all_possible_hidden_states)):
                for l in range(len(all_possible_hidden_states)):
                    iteration_sum[l]=intermediate[k]*transition_matrix[k][l]
                for m in range(len(all_possible_hidden_states)):
                    final_sum[m]=final_sum[m]+iteration_sum[m]
        else:
            x,y=observations[i-1]
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
        forward_messages[i]=[message]
    
    backward_messages = [None] * num_time_steps
    backward_messages[num_time_steps-1]=[prior_backward]
    # TODO: Compute the backward messages\
    for i in range(num_time_steps-1,0,-1):
        iteration_sum=[0] * len(all_possible_hidden_states)
        intermediate = [None] * len(all_possible_hidden_states)
        final_sum=[0] * len(all_possible_hidden_states)
        message=robot.Distribution()
        if observations[i]==None:
            index=0
            for j in range(len(all_possible_hidden_states)):
                intermediate[j]=backward_messages[i][0][all_possible_hidden_states[j]]*none_observation_matrix[j][index]
            for k in range(len(all_possible_hidden_states)):
                for l in range(len(all_possible_hidden_states)):
                        iteration_sum[l]=intermediate[k]*back_transition_matrix[k][l]
                for m in range(len(all_possible_hidden_states)):
                        final_sum[m]=final_sum[m]+iteration_sum[m]
        else:
            x,y=observations[i]
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
        if observations[i]==None:
            index=0
            if i>0 and i<num_time_steps:
                for j in range(len(all_possible_hidden_states)):
                    marginal_distribution[all_possible_hidden_states[j]]=none_observation_matrix[j][index]*backward_messages[i][0][all_possible_hidden_states[j]]*forward_messages[i][0][all_possible_hidden_states[j]]
            if i==0:
                for j in range(len(all_possible_hidden_states)):
                    marginal_distribution[all_possible_hidden_states[j]]=none_observation_matrix[j][index]*backward_messages[0][0][all_possible_hidden_states[j]]
            elif i==num_time_steps-1:
                for j in range(len(all_possible_hidden_states)):
                    marginal_distribution[all_possible_hidden_states[j]]=none_observation_matrix[j][index]*forward_messages[num_time_steps-1][0][all_possible_hidden_states[j]]   
        else:
            x,y=observations[i]
            index=all_possible_observed_states.index((x,y))
            if i>0 and i<num_time_steps:
                for j in range(len(all_possible_hidden_states)):
                    marginal_distribution[all_possible_hidden_states[j]]=observation_matrix[j][index]*backward_messages[i][0][all_possible_hidden_states[j]]*forward_messages[i][0][all_possible_hidden_states[j]]
            if i==0:
                for j in range(len(all_possible_hidden_states)):
                    marginal_distribution[all_possible_hidden_states[j]]=observation_matrix[j][index]*backward_messages[0][0][all_possible_hidden_states[j]]
            if i==num_time_steps-1:
                for j in range(len(all_possible_hidden_states)):
                    marginal_distribution[all_possible_hidden_states[j]]=observation_matrix[j][index]*forward_messages[num_time_steps-1][0][all_possible_hidden_states[j]]   
        marginals[i]=marginal_distribution
        marginals[i].renormalize()
    return marginals


def Viterbi(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #


    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps # remove this

    return estimated_hidden_states


def second_best(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #


    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps # remove this

    return estimated_hidden_states


# -----------------------------------------------------------------------------
# Generating data from the hidden Markov model
#

def generate_data(num_time_steps, make_some_observations_missing=False,
                  random_seed=None):
    # generate samples from this project's hidden Markov model
    hidden_states = []
    observations = []

    # if the random seed is not None, then this makes the randomness
    # deterministic, which may be helpful for debug purposes
    np.random.seed(random_seed)

    # draw initial state and emit an observation
    initial_state = prior_distribution.sample()
    initial_observation = observation_model(initial_state).sample()

    hidden_states.append(initial_state)
    observations.append(initial_observation)

    for time_step in range(1, num_time_steps):
        # move the robot
        prev_state = hidden_states[-1]
        new_state = transition_model(prev_state).sample()

        # maybe emit an observation
        if not make_some_observations_missing:
            new_observation = observation_model(new_state).sample()
        else:
            if np.random.rand() < .1:  # 0.1 prob. of observation being missing
                new_observation = None
            else:
                new_observation = observation_model(new_state).sample()

        hidden_states.append(new_state)
        observations.append(new_observation)

    return hidden_states, observations


# -----------------------------------------------------------------------------
# Main
#

def main():
    # flags
    make_some_observations_missing = False
    use_graphics = True
    need_to_generate_data = True

    # parse command line arguments
    for arg in sys.argv[1:]:
        if arg == '--missing':
            make_some_observations_missing = True
        elif arg == '--nographics':
            use_graphics = False
        elif arg.startswith('--load='):
            filename = arg[7:]
            hidden_states, observations = robot.load_data(filename)
            need_to_generate_data = False
            num_time_steps = len(hidden_states)

    # if no data is loaded, then generate new data
    if need_to_generate_data:
        num_time_steps = 100
        hidden_states, observations = \
            generate_data(num_time_steps,
                          make_some_observations_missing)
    print('Running forward-backward...')
    marginals = forward_backward([(4, 3), (4, 2), (3, 2), (4, 0), (2, 0), (2, 0), (3, 2), (4, 2), (2, 3), (3, 5)])
    print("\n")

    timestep = 2
    print("Most likely parts of marginal at time %d:" % (timestep))
    if marginals[timestep] is not None:
        print(sorted(marginals[timestep].items(),
                     key=lambda x: x[1],
                     reverse=True)[:10])
    else:
        print('*No marginal computed*')
    print("\n")

    print('Running Viterbi...')
    estimated_states = Viterbi(observations)
    print("\n")

    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states[time_step] is None:
            print('Missing')
        else:
            print(estimated_states[time_step])
    print("\n")

    print('Finding second-best MAP estimate...')
    estimated_states2 = second_best(observations)
    print("\n")

    print("Last 10 hidden states in the second-best MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states2[time_step] is None:
            print('Missing')
        else:
            print(estimated_states2[time_step])
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != hidden_states[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between MAP estimate and true hidden " +
          "states:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states2[time_step] != hidden_states[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between second-best MAP estimate and " +
          "true hidden states:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != estimated_states2[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between MAP and second-best MAP " +
          "estimates:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    # display
    if use_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()


if __name__ == '__main__':
    main()
