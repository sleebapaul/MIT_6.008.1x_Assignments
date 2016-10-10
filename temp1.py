# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 12:20:24 2016
Topic : How to create and represent joint probablity distributions 
@author: sleeba
""" 
import pandas as pd
#approach 1
prob_table = {('sunny', 'hot'): 3/10,('sunny', 'cold'): 1/5,('rainy', 'hot'): 1/30,('rainy', 'cold'): 2/15,('snowy', 'hot'): 0,('snowy', 'cold'): 1/3}
print (prob_table)

#appraoch 2
prob_W_T_dict = {}
for w in {'sunny', 'rainy', 'snowy'}:
    prob_W_T_dict[w] = {}
prob_W_T_dict['sunny']['hot'] = 1/4
prob_W_T_dict['sunny']['cold'] = 1/4
prob_W_T_dict['rainy']['hot'] = 1/12
prob_W_T_dict['rainy']['cold'] = 2/12
prob_W_T_dict['snowy']['hot'] = 1/6
prob_W_T_dict['snowy']['cold'] = 1/6
print (prob_W_T_dict)
print(pd.DataFrame(prob_W_T_dict).T)

#approach 3
import numpy as np
prob_W_T_rows = ['sunny', 'rainy', 'snowy']
prob_W_T_cols = ['hot', 'cold']
prob_W_T_array = np.array([[3/10, 1/5], [1/30, 2/15], [0, 1/3]])
print(pd.DataFrame(prob_W_T_array,prob_W_T_rows, prob_W_T_cols))
print ( prob_W_T_array[prob_W_T_rows.index('rainy'), prob_W_T_cols.index('cold')])

'''Using .index does a search through the whole list of row/column labels, 
which for large lists can be slow. A cleaner and faster way is to create 
separate dictionaries mapping the row and column labels 
to row and column indices in the 2D array.'''
prob_W_T_row_mapping = {}
for index, label in enumerate(prob_W_T_rows):
    prob_W_T_row_mapping[label] = index
#or using dictionary comprehension
prob_W_T_rows = ['sunny', 'rainy', 'snowy']
prob_W_T_cols = ['hot', 'cold']
prob_W_T_row_mapping = {key: value for key, value in enumerate(prob_W_T_rows)}
prob_W_T_col_mapping = {key: value for key, value in enumerate(prob_W_T_cols)}
prob_W_T_array = np.array([[3/10, 1/5], [1/30, 2/15], [0, 1/3]])
#print (prob_W_T_array[prob_W_T_row_mapping[w], prob_W_T_col_mapping[t]])
print (prob_W_T_row_mapping)
love={'sunny':1/2,'rainy':1/6,'snowy':1/3}