# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 00:42:44 2016

@author: sleeba
"""
import numpy as np
#prob_W_I = np.array([[1/2, 0], [0, 1/6], [0, 1/3]])
#prob_W = prob_W_I.sum(axis=1)
#prob_I = prob_W_I.sum(axis=0)
#print (prob_W)
#print (prob_I)
#print(np.outer(prob_W, prob_I))

prob_X_Y = np.array([[1/4, 1/4], [1/12, 1/12], [1/6, 1/6]])
prob_X = prob_X_Y.sum(axis=1)
prob_Y = prob_X_Y.sum(axis=0)
print (prob_X)
print (prob_Y)
print(np.outer(prob_X, prob_Y))
print (prob_X_Y)

