## -*- coding: utf-8 -*-
#"""
#Created on Fri Nov 11 12:47:54 2016
#
#@author: sleeba
#"""
#
#import numpy as np
#import util
#file_list=get_files_in_folder('data\ham')
##words=[[] for i in range(2)]
##print (words)
##words[0]=['stay','stay','up']
##words[1]=['down','left','stay','left']
##flattened_list=[item for sublist in words for item in sublist]
##print (flattened_list)
#def get_counts(file_list):
#    """
#    Computes counts for each word that occurs in the files in file_list.
#
#    Inputs
#    ------
#    file_list : a list of filenames, suitable for use with open() or 
#                util.get_words_in_file()
#
#    Output
#    ------
#    A dict whose keys are words, and whose values are the number of files the
#    key occurred in.
#    """
#    ### TODO: Comment out the following line and write your code here
#    #raise NotImplementedError
#    words=[[] for i in range(len(file_list))]
#    out={}
#    for i in range(len(file_list)):
#        file_name=file_list[i]
#        words[i]=util.get_words_in_file(file_name)
#        words[i]=np.unique(words[i])
#    flattened_list=[item for sublist in words for item in sublist]
#    flattened_list=np.unique(flattened_list)
#    for i in range(len(flattened_list)):
#        count=0
#        for j in range(len(file_list)):
#            if flattened_list[i] in words[j]:
#                count=count+1
#        out[flattened_list[i]]=count
#    return out
#def get_log_probabilities(file_list):
#    """
#    Computes log-frequencies for each word that occurs in the files in 
#    file_list.
#
#    Input
#    -----
#    file_list : a list of filenames, suitable for use with open() or 
#                util.get_words_in_file()
#
#    Output
#    ------
#    A dict whose keys are words, and whose values are the log of the smoothed
#    estimate of the fraction of files the key occurred in.
#
#    Hint
#    ----
#    The data structure util.DefaultDict will be useful to you here, as will the
#    get_counts() helper above.
#    """
#    ### TODO: Comment out the following line and write your code here
#    #raise NotImplementedError
#    out=get_counts (file_list)
#    out_log={}
#    for i in out:
#        out_log[i]=np.log(out[i])
#    return out_log
#print(get_log_probabilities(file_list))
print ((1+2)/(2+1))