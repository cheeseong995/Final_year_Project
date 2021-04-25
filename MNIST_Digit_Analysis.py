# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:55:01 2021

@author: User
"""

# Load all the Dictionaries
import pickle
import matplotlib.pyplot as plt
import master_code_final_v1 as mcf1
import numpy as np
from sklearn import metrics

file_to_read = open("digit_gd_05.pickle" , "rb")
digit_gd_05 = pickle.load(file_to_read)

file_to_read = open("digit_gd_armijo.pickle" , "rb")
digit_gd_armijo = pickle.load(file_to_read)

file_to_read = open("digit_gd_decay.pickle" , "rb")
digit_gd_decay = pickle.load(file_to_read)

file_to_read = open("digit_momentum_05.pickle" , "rb")
digit_mom_05 = pickle.load(file_to_read)

file_to_read = open("digit_momentum_armijo.pickle" , "rb")
digit_mom_armijo = pickle.load(file_to_read)

file_to_read = open("digit_momentum_decay.pickle" , "rb")
digit_mom_decay = pickle.load(file_to_read)

# --------------------------------------------------------------------------------------

plt.plot(np.squeeze(digit_gd_05["accuracy_list_train"]) , label = "gd_05")
plt.plot(np.squeeze(digit_gd_armijo["accuracy_list_train"]) , label = "gd_armijo")
plt.plot(np.squeeze(digit_mom_05["accuracy_list_train"]) , label = "mom_05")
plt.plot(np.squeeze(digit_mom_armijo["accuracy_list_train"]) , label = "mom_armijo")
plt.ylim(top = 1 , bottom = 0.95)
plt.xlim(0,400)
locs,labels = plt.xticks()
plt.xticks(locs,[int(x) for x in (locs*5)])
plt.ylabel('Training Accuracy')
plt.xlabel('Iterations')
plt.title("Training Accuracy")
plt.legend(loc="upper left right",shadow = True)
plt.show()

plt.plot(np.squeeze(digit_gd_05["accuracy_list_valid"]) , label = "gd_05")
plt.plot(np.squeeze(digit_gd_armijo["accuracy_list_valid"]) , label = "gd_armijo")
plt.plot(np.squeeze(digit_mom_05["accuracy_list_valid"]) , label = "mom_05")
plt.plot(np.squeeze(digit_mom_armijo["accuracy_list_valid"]) , label = "mom_armijo")
plt.ylim(top = 1 , bottom = 0.95)
plt.xlim(0,400)
locs,labels = plt.xticks()
plt.xticks(locs,[int(x) for x in (locs*5)])
plt.ylabel('Validation Accuracy')
plt.xlabel('Iterations')
plt.title("Validation Accuracy")
plt.legend(loc="upper left",shadow = True)
plt.show()

# ------------------------------------------------------------------------------------------

plt.plot(np.squeeze(digit_gd_armijo["learning_rates"]) , label = "gd_armijo")
plt.plot(np.squeeze(digit_gd_05["learning_rates"]) , label = "gd_05" )
plt.plot(np.squeeze(digit_gd_decay["learning_rates"]) , label = "gd_decay")
plt.ylabel('Learning Rates')
plt.xlabel('Iteration')
plt.title("Learning Rates")
plt.legend(loc="upper right",shadow = True)
plt.show()

plt.plot(np.squeeze(digit_mom_armijo["learning_rates"]) , label = "mom_armijo")
plt.plot(np.squeeze(digit_mom_05["learning_rates"]) , label = "mom_05")
plt.plot(np.squeeze(digit_mom_decay["learning_rates"]) , label = "mom_decay")
plt.ylabel('Learning Rates')
plt.xlabel('Iteration')
plt.title("Learning Rates")
plt.legend(loc="upper right",shadow = True)
plt.show()

# ------------------------------------------------------------------------------------------
print("")
print("Training Accuracy of GD for fixed learning rate:" , digit_gd_05["accuracy_train"])
print("Test Accuracy of GD for fixed learning rate:" , digit_gd_05["accuracy_test"])
print("")
print("Training Accuracy of GD for decay learning rate:" , digit_gd_decay["accuracy_train"])
print("Test Accuracy of GD for decay learning rate:" , digit_gd_decay["accuracy_test"])
print("")
print("Training Accuracy of GD for armijo learning rate:" , digit_gd_armijo["accuracy_train"])
print("Test Accuracy of GD for armijo learning rate:" , digit_gd_armijo["accuracy_test"])
print("")
print("Training Accuracy of GDM for fixed learning rate:" , digit_mom_05["accuracy_train"])
print("Test Accuracy of GDM for fixed learning rate:" , digit_mom_05["accuracy_test"])
print("")
print("Training Accuracy of GDM for decay learning rate:" , digit_mom_decay["accuracy_train"])
print("Test Accuracy of GDM for decay learning rate:" , digit_mom_decay["accuracy_test"])
print("")
print("Training Accuracy of GDM for armijo learning rate:" , digit_mom_armijo["accuracy_train"])
print("Test Accuracy of GDM for armijo learning rate:" , digit_mom_armijo["accuracy_test"])

# ------------------------------------------------------------------------------------------
print("")
print("Time taken of GD for fixed learning rate" , digit_gd_05["time"] )
print("Iteration of GD for fixed learning rate" , digit_gd_05["num_iteration"] )
print("")
print("Time taken of GD for decay learning rate" , digit_gd_decay["time"] )
print("Iteration of GD for decay learning rate" , digit_gd_decay["num_iteration"] )
print("")
print("Time taken of GD for armijo learning rate" , digit_gd_armijo["time"] )
print("Iteration of GD for armijo learning rate" , digit_gd_armijo["num_iteration"] )
print("")
print("Time taken of GDM for fixed learning rate" , digit_mom_05["time"] )
print("Iteration of GDM for fixed learning rate" , digit_mom_05["num_iteration"] )
print("")
print("Time taken of GDM for decay learning rate" , digit_mom_decay["time"] )
print("Iteration of GDM for decay learning rate" , digit_mom_decay["num_iteration"] )
print("")
print("Time taken of GDM for armijo learning rate" , digit_mom_armijo["time"] )
print("Iteration of GDM for armijo learning rate" , digit_mom_armijo["num_iteration"] )

# ------------------------------------------------------------------------------------------

X_train,Y_train,X_test,Y_test = mcf1.get_data("digit")
Y_test_labels = np.argmax(Y_test, axis=0)
print("")
print("Gradient Descent with fixed learning rate")
Y_prediction_test = mcf1.predict(digit_gd_05["parameters"], X_test)
Y_test_labels_gd = np.argmax(Y_prediction_test, axis=0)
print("Accuracy:" , np.round(metrics.accuracy_score(Y_test_labels,Y_test_labels_gd),9))
print("Precision:" , np.round(metrics.precision_score(Y_test_labels,Y_test_labels_gd,average = "weighted"),9))
print("Recall:" , np.round(metrics.recall_score(Y_test_labels,Y_test_labels_gd,average = "weighted"),9))
print("F1-score:" , np.round(metrics.f1_score(Y_test_labels,Y_test_labels_gd,average = "weighted"),9))
print("")
print("Gradient Descent with armijo learning rate")
Y_prediction_test = mcf1.predict(digit_gd_armijo["parameters"], X_test)
Y_test_labels_gd = np.argmax(Y_prediction_test, axis=0)
print("Accuracy:" , np.round(metrics.accuracy_score(Y_test_labels,Y_test_labels_gd),5))
print("Precision:" , np.round(metrics.precision_score(Y_test_labels,Y_test_labels_gd,average = "weighted"),5))
print("Recall:" , np.round(metrics.recall_score(Y_test_labels,Y_test_labels_gd,average = "weighted"),6))
print("F1-score:" , np.round(metrics.f1_score(Y_test_labels,Y_test_labels_gd,average = "weighted"),5))
print("")
print("Gradient Descent with decay learning rate")
Y_prediction_test = mcf1.predict(digit_gd_decay["parameters"], X_test)
Y_test_labels_gd = np.argmax(Y_prediction_test, axis=0)
print("Accuracy:" , np.round(metrics.accuracy_score(Y_test_labels,Y_test_labels_gd),5))
print("Precision:" , np.round(metrics.precision_score(Y_test_labels,Y_test_labels_gd,average = "weighted"),5))
print("Recall:" , np.round(metrics.recall_score(Y_test_labels,Y_test_labels_gd,average = "weighted"),6))
print("F1-score:" , np.round(metrics.f1_score(Y_test_labels,Y_test_labels_gd,average = "weighted"),5))
print("")
print("Gradient Descent Mom with fixed learning rate")
Y_prediction_test = mcf1.predict(digit_mom_05["parameters"], X_test)
Y_test_labels_gd = np.argmax(Y_prediction_test, axis=0)
print("Accuracy:" , np.round(metrics.accuracy_score(Y_test_labels,Y_test_labels_gd),5))
print("Precision:" , np.round(metrics.precision_score(Y_test_labels,Y_test_labels_gd,average = "weighted"),5))
print("Recall:" , np.round(metrics.recall_score(Y_test_labels,Y_test_labels_gd,average = "weighted"),6))
print("F1-score:" , np.round(metrics.f1_score(Y_test_labels,Y_test_labels_gd,average = "weighted"),5))
print("")
print("Gradient Descent Mom with armijo learning rate")
Y_prediction_test = mcf1.predict(digit_mom_armijo["parameters"], X_test)
Y_test_labels_gd = np.argmax(Y_prediction_test, axis=0)
print("Accuracy:" , np.round(metrics.accuracy_score(Y_test_labels,Y_test_labels_gd),5))
print("Precision:" , np.round(metrics.precision_score(Y_test_labels,Y_test_labels_gd,average = "weighted"),5))
print("Recall:" , np.round(metrics.recall_score(Y_test_labels,Y_test_labels_gd,average = "weighted"),6))
print("F1-score:" , np.round(metrics.f1_score(Y_test_labels,Y_test_labels_gd,average = "weighted"),5))
print("")
print("Gradient Descent Mom with decay learning rate")
Y_prediction_test = mcf1.predict(digit_mom_decay["parameters"], X_test)
Y_test_labels_gd = np.argmax(Y_prediction_test, axis=0)
print("Accuracy:" , np.round(metrics.accuracy_score(Y_test_labels,Y_test_labels_gd),5))
print("Precision:" , np.round(metrics.precision_score(Y_test_labels,Y_test_labels_gd,average = "weighted"),5))
print("Recall:" , np.round(metrics.recall_score(Y_test_labels,Y_test_labels_gd,average = "weighted"),6))
print("F1-score:" , np.round(metrics.f1_score(Y_test_labels,Y_test_labels_gd,average = "weighted"),5))

