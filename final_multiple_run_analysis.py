# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 16:36:17 2021

@author: User
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt

file_to_read = open("models_50_gd_05.pickle" , "rb")
models_50_gd_05 = pickle.load(file_to_read)

for i in range(1,51):
    plt.plot(models_50_gd_05["gs_cost" + str(i)])

models_50_gd_05_training_acc = []
models_50_gd_05_test_acc = []
models_50_gd_05_iterations = []
models_50_gd_05_time = []

for i in range(1,51):
    
    models_50_gd_05_training_acc.append(models_50_gd_05["Model" + str(i)]["accuracy_train"])
    models_50_gd_05_test_acc.append(models_50_gd_05["Model" + str(i)]["accuracy_test"])
    models_50_gd_05_iterations.append(models_50_gd_05["Model" + str(i)]["num_iteration"])
    time_taken = models_50_gd_05["Model" + str(i)]["time"] + models_50_gd_05["gs_time" + str(i)]
    models_50_gd_05_time.append(time_taken)

print("-------------------------------------------")
print("")
print("Metrics for digits: Gradient descent with learning rate 0.5")
print("The mean for training accuracy is:", np.round(np.mean(models_50_gd_05_training_acc),4))
print("The sd for training accuracy is:", np.round(np.std(models_50_gd_05_training_acc),4))
print("The mean for test accuracy is:", np.round(np.mean(models_50_gd_05_test_acc),4))
print("The sd for test accuracy is:", np.round(np.std(models_50_gd_05_test_acc),4))
print("The mean for iterations is:", np.round(np.mean(models_50_gd_05_iterations),4))
print("The sd for iterations is:", np.round(np.std(models_50_gd_05_iterations),4))
print("The mean for time is:", np.round(np.mean(models_50_gd_05_time),4))
print("The sd for time is:", np.round(np.std(models_50_gd_05_time),4))
print("")

# ------------------------------------------------------------------------------------------------------------

file_to_read = open("models_50_gd_armijo.pickle" , "rb")
models_50_gd_armijo = pickle.load(file_to_read)

models_50_gd_armijo_training_acc = []
models_50_gd_armijo_test_acc = []
models_50_gd_armijo_iterations = []
models_50_gd_armijo_time = []

for i in range(1,51):
    
    models_50_gd_armijo_training_acc.append(models_50_gd_armijo["Model" + str(i)]["accuracy_train"])
    models_50_gd_armijo_test_acc.append(models_50_gd_armijo["Model" + str(i)]["accuracy_test"])
    models_50_gd_armijo_iterations.append(models_50_gd_armijo["Model" + str(i)]["num_iteration"])
    time_taken = models_50_gd_armijo["Model" + str(i)]["time"]
    models_50_gd_armijo_time.append(time_taken)

print("-------------------------------------------")
print("")
print("Metrics for digits: Gradient descent with armijo learning rate")
print("The mean for training accuracy is:", np.round(np.mean(models_50_gd_armijo_training_acc),4))
print("The sd for training accuracy is:", np.round(np.std(models_50_gd_armijo_training_acc),4))
print("The mean for test accuracy is:", np.round(np.mean(models_50_gd_armijo_test_acc),4))
print("The sd for test accuracy is:", np.round(np.std(models_50_gd_armijo_test_acc),4))
print("The mean for iterations is:", np.round(np.mean(models_50_gd_armijo_iterations),4))
print("The sd for iterations is:", np.round(np.std(models_50_gd_armijo_iterations),4))
print("The mean for time is:", np.round(np.mean(models_50_gd_armijo_time),4))
print("The sd for time is:", np.round(np.std(models_50_gd_armijo_time),4))
print("")

# ------------------------------------------------------------------------------------------------------------

file_to_read = open("models_50_mom_05.pickle" , "rb")
models_50_mom_05 = pickle.load(file_to_read)

for i in range(1,51):
    plt.plot(models_50_mom_05["gs_cost" + str(i)])

models_50_mom_05_training_acc = []
models_50_mom_05_test_acc = []
models_50_mom_05_iterations = []
models_50_mom_05_time = []

for i in range(1,51):
    
    models_50_mom_05_training_acc.append(models_50_mom_05["Model" + str(i)]["accuracy_train"])
    models_50_mom_05_test_acc.append(models_50_mom_05["Model" + str(i)]["accuracy_test"])
    models_50_mom_05_iterations.append(models_50_mom_05["Model" + str(i)]["num_iteration"])
    time_taken = models_50_mom_05["Model" + str(i)]["time"] + models_50_mom_05["gs_time" + str(i)]
    models_50_mom_05_time.append(time_taken)

print("-------------------------------------------")
print("")
print("Metrics for digits: Gradient descent with momentum with learning rate 0.5")
print("The mean for training accuracy is:", np.round(np.mean(models_50_mom_05_training_acc),4))
print("The sd for training accuracy is:", np.round(np.std(models_50_mom_05_training_acc),4))
print("The mean for test accuracy is:", np.round(np.mean(models_50_mom_05_test_acc),4))
print("The sd for test accuracy is:", np.round(np.std(models_50_mom_05_test_acc),4))
print("The mean for iterations is:", np.round(np.mean(models_50_mom_05_iterations),4))
print("The sd for iterations is:", np.round(np.std(models_50_mom_05_iterations),4))
print("The mean for time is:", np.round(np.mean(models_50_mom_05_time),4))
print("The sd for time is:", np.round(np.std(models_50_mom_05_time),4))
print("")

# ------------------------------------------------------------------------------------------------------------
file_to_read = open("models_50_mom_armijo.pickle" , "rb")
models_50_mom_armijo = pickle.load(file_to_read)

models_50_mom_armijo_training_acc = []
models_50_mom_armijo_test_acc = []
models_50_mom_armijo_iterations = []
models_50_mom_armijo_time = []

for i in range(1,51):
    
    models_50_mom_armijo_training_acc.append(models_50_mom_armijo["Model" + str(i)]["accuracy_train"])
    models_50_mom_armijo_test_acc.append(models_50_mom_armijo["Model" + str(i)]["accuracy_test"])
    models_50_mom_armijo_iterations.append(models_50_mom_armijo["Model" + str(i)]["num_iteration"])
    time_taken = models_50_mom_armijo["Model" + str(i)]["time"]
    models_50_mom_armijo_time.append(time_taken)

print("-------------------------------------------")
print("")
print("Metrics for digits: Gradient descent with momentum with armijo learning rate")
print("The mean for training accuracy is:", np.round(np.mean(models_50_mom_armijo_training_acc),4))
print("The sd for training accuracy is:", np.round(np.std(models_50_mom_armijo_training_acc),4))
print("The mean for test accuracy is:", np.round(np.mean(models_50_mom_armijo_test_acc),4))
print("The sd for test accuracy is:", np.round(np.std(models_50_mom_armijo_test_acc),4))
print("The mean for iterations is:", np.round(np.mean(models_50_mom_armijo_iterations),4))
print("The sd for iterations is:", np.round(np.std(models_50_mom_armijo_iterations),4))
print("The mean for time is:", np.round(np.mean(models_50_mom_armijo_time),4))
print("The sd for time is:", np.round(np.std(models_50_mom_armijo_time),4))
print("")

    