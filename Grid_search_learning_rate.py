# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 20:48:28 2021

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import master_code_final_v1_seed as mcf1
import time

X_train,Y_train,X_test,Y_test = mcf1.get_data("fashion")
n_x = X_train.shape[0]
n_y = Y_train.shape[0]
layer_dims = [n_x,260,n_y]

def grid_search(start , stop , steps , typ , X_train , Y_train , X_test , Y_test , layer_dims):
    
    time_start = time.time()
    
    if typ == "expo":
        diff = np.exp(np.log(stop/start)/steps)
    elif typ == "constant":
        diff = (stop-start)/steps

    
    learning_rates = []
    costs_learning_rates = []
    temp_learning_rate = start
    for i in range(steps):
        learning_rates.append(temp_learning_rate)
        if typ =="expo":
            temp_learning_rate *=diff
        elif typ == "constant":
            temp_learning_rate +=diff
    
    for i in learning_rates:
        model = mcf1.model(X_train ,Y_train, X_test, Y_test, layer_dims, optimizer = "gd", 
          learning_rate = i , num_iter = 20 , batch_size = 1024, decay = False, print_cost = False)
        costs_learning_rates.append(model["costs"][-1])
     
    time_stop = time.time()
    total_time = time_stop - time_start
    
    return total_time , learning_rates , costs_learning_rates

start = 0.01
stop = 1
steps = 20
total_time1 , learning_rates1, costs1 = grid_search(start , stop , steps , "expo" , 
                                          X_train , Y_train , X_test , Y_test , layer_dims)

start = 0.05
stop = 0.5
steps = 20
total_time2 , learning_rates2, costs2 = grid_search(start , stop , steps , "constant" , 
                                          X_train , Y_train , X_test , Y_test , layer_dims)
# -----------------------------------------------------
rangearc = []
for i in range(10):
    rangearc.append(round(learning_rates1[i*2],3))
    
plt.plot(costs1)
plt.xlabel("Learning rate (log)")
plt.ylabel("costs")
plt.xlim(0,20)
plt.locator_params(axis = "x" , nbins = 10)
plt.title("1st Grid-Search for learning rate")
loc , label = plt.xticks()
plt.xticks(loc,rangearc)
plt.axvline(8 , linestyle = "--" , color = "darkorange")
plt.axvline(13 , linestyle = "--" , color = "darkorange")


rangearc = []
for i in range(10):
    rangearc.append(round(learning_rates2[i*2],3))
    
plt.plot(costs2)
plt.xlabel("Learning rate")
plt.ylabel("costs")
plt.xlim(0,20)
plt.locator_params(axis = "x" , nbins = 10)
plt.title("2nd Grid-Search for learning rate")
loc , label = plt.xticks()
plt.xticks(loc,rangearc)
plt.axvline(2 , linestyle = "--" , color = "darkorange")
plt.axvline(7 , linestyle = "--" , color = "darkorange")
# -----------------------------------------------------

timefor = total_time1 + total_time2
