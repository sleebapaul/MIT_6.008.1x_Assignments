# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 10:53:28 2016

@author: sleeba
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def sample_from_finite_probability_space(finite_prob_space):
    """
    Produces a random outcome from a given finite probability space.

    Input
    -----
    - finite_prob_space: finite probability space encoded as a
      dictionary

    Output
    ------
    - random outcome, which is one of the keys in the
      finite_probability_space dictionary's set of keys
      (remember: these keys form the sample space)
    """

    # first produce a list of pairs of the form (outcome, outcome probability)
    outcome_probability_pairs = list(finite_prob_space.items())

    # convert the pairs into two lists "outcomes" and "outcome_probabilities":
    # - outcomes: list of outcomes
    # - outcome_probabilities: i-th element is the probability of the i-th
    #   outcome in the "outcomes" list
    # (note that this step is needed because NumPy wants these lists
    # separately)
    outcomes, outcome_probabilities = zip(*outcome_probability_pairs)

    # use NumPy to randomly sample
    random_outcome = np.random.choice(outcomes, p=outcome_probabilities)
    return random_outcome


def flip_fair_coin():
    """
    Returns a fair coin flip.

    Output
    ------
    - either the string 'heads' or 'tails'
    """
    finite_prob_space = {'heads': 0.5, 'tails': 0.5}
    return sample_from_finite_probability_space(finite_prob_space)
#n = 100000
#heads_so_far = 0
#fraction_of_heads = []
#for i in range(n):
#    if flip_fair_coin() == 'heads':
#        heads_so_far += 1
#    fraction_of_heads.append(heads_so_far / (i+1))
#plt.figure(figsize=(8, 4))
#plt.plot(range(1, n+1), fraction_of_heads)
#plt.xlabel('Number of flips')
#plt.ylabel('Fraction of heads')
prob_space = {'sunny': 1/2, 'rainy': 1/6, 'snowy': 1/3}
#W_mapping = {'sunny': 'sunny', 'rainy': 'rainy', 'snowy': 'snowy'}
#I_mapping = {'sunny': 1, 'rainy': 0, 'snowy': 0}
#random_outcome =sample_from_finite_probability_space(prob_space)
#print (random_outcome)
#W = W_mapping[random_outcome]
#I = I_mapping[random_outcome]
#print (W)
#print (I)
W_table = {'sunny': 1/2, 'rainy': 1/6, 'snowy': 1/3}
I_table = {0: 1/2, 1: 1/2}
W =sample_from_finite_probability_space(W_table)
I =sample_from_finite_probability_space(I_table)
print (W)
print (I)