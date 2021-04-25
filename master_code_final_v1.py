"""
Master code for my final year project
Adaptive Learning rate for Neural Network
@author: Chee Seong
Date started: June 2020
Date Ended: April 2020
"""

#import libraries
import numpy as np
import random
import tensorflow as tf
from keras.utils import np_utils
import time
import matplotlib.pyplot as plt
import math


def get_data(dat_set):    
    if dat_set == "digit":
        (train_set_x,train_set_y),(test_set_x,test_set_y) = tf.keras.datasets.mnist.load_data()
    if dat_set == "fashion":
        (train_set_x,train_set_y),(test_set_x,test_set_y) = tf.keras.datasets.fashion_mnist.load_data()
    train_set_y = np_utils.to_categorical(train_set_y, 10) #One-hot encoded
    test_set_y= np_utils.to_categorical(test_set_y, 10) #One-hot encoded
    train_set_x_flatten = train_set_x.reshape(train_set_x.shape[0] , -1).T #Flatten it and transpose to a usable format
    test_set_x_flatten =  test_set_x.reshape(test_set_x.shape[0] , -1).T #Flatten it and transpose to a usable format
    
    X_train = train_set_x_flatten/255.    #Normalize the data set
    X_test = test_set_x_flatten/255.
    Y_train = train_set_y.T              # Transpose it so that it fits the format for X_train and X_test 
    Y_test = test_set_y.T
     
    return X_train,Y_train,X_test,Y_test

def random_mini_batches(X,Y,batch_size=64):
    m = X.shape[1]
    random.seed()
    sampling = random.sample(range(1,m),batch_size)
    
    mini_batch_X = X[:, sampling]
    mini_batch_Y = Y[:, sampling]
 
    return mini_batch_X , mini_batch_Y

def get_validation_data(X,Y,seed):
    np.random.seed(seed)
    m = X.shape[1]
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((10,m))
    
    X_train = shuffled_X[:,0:50000]
    Y_train = shuffled_Y[:,0:50000]
    data_train = (X_train,Y_train)
    
    X_train_valid = shuffled_X[:,50000:60000]
    Y_train_valid = shuffled_Y[:,50000:60000]
    data_valid = (X_train_valid,Y_train_valid)
    
    return data_train,data_valid

def plot_image(i,X_train,Y_train):
    plt.imshow(X_train[i])
    plt.title(Y_train[i])

def softmax(z): 
    return np.exp(z)/np.sum(np.exp(z),axis=0)

def relu(z): 
    return np.maximum(0,z)

def initialize(layer_dims): 

    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1,L):
        if l ==(L-1):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*np.sqrt(1/layer_dims[l-1])
            parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
        else:
            parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*np.sqrt(2/layer_dims[l-1])
            parameters['b' + str(l)] = np.zeros((layer_dims[l],1))

    return parameters 

def initialize_velocity(parameters): 
    
    L = len(parameters) // 2 
    v = {}
    
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros((parameters['W'+str(l+1)].shape[0] , parameters['W'+str(l+1)].shape[1]))
        v["db" + str(l+1)] = np.zeros((parameters['b'+str(l+1)].shape[0] , parameters['b'+str(l+1)].shape[1]))
    
    return v

def forward_propagation(parameters,X):
    
    L = len(parameters) // 2  
    A = {}
    z = {}
    A["A" + str(0)] = X
    
    for l in range(L):
        if l == L-1:  # Use the softmax/sigmoid function
            z["z" + str(l+1)] = np.dot(parameters["W" + str(l+1)],A["A" + str(l)]) + parameters["b" + str(l+1)]
            A["A" + str(l+1)] = softmax(z["z" + str(l+1)])
            last_A = A["A" + str(l+1)]                            

        else: # Else use the Relu function
            z["z" + str(l+1)] = np.dot(parameters["W" + str(l+1)],A["A" + str(l)]) + parameters["b" + str(l+1)]
            A["A" + str(l+1)] = relu(z["z" + str(l+1)])
            

    return last_A , A , z

def compute_loss(last_A , parameters, Y, lambd):
    
    L = len(parameters) // 2
    total_sum = 0
    
    temp_cost = -(np.multiply(np.log(last_A),Y))

    
    for l in range(L):
        total_sum += np.sum(np.square(parameters["W"+str(l+1)]))
    L2_regularization_cost = (lambd/2)*total_sum
    cost = (np.sum(temp_cost)+L2_regularization_cost)
   
    return cost

def backward_propagation(parameters, A , z, X ,Y,lambd):
    
    L = len(parameters) // 2
    m = X.shape[1]
    grads = {}
    
    for l in range(L,0,-1):
        if l == L:
            grads["dZ"+str(l)] = A["A" + str(l)] - Y
            grads["dW"+str(l)] = (1/m)*np.dot(grads["dZ"+str(l)],A["A"+str(l-1)].T) + (lambd/m*parameters["W"+str(l)])
            grads["db"+str(l)] = (1/m)*np.sum(grads["dZ"+str(l)],axis=1,keepdims=True)
        else:
            grads["dA"+str(l)] = np.dot(parameters["W"+str(l+1)].T,grads["dZ"+str(l+1)])
            grads["dZ"+str(l)] = np.multiply(grads["dA"+str(l)],np.int64(A["A"+str(l)]>0))
            grads["dW"+str(l)] = (1/m)*np.dot(grads["dZ"+str(l)],A["A"+str(l-1)].T) + (lambd/m*parameters["W"+str(l)])
            grads["db"+str(l)] = (1/m)*np.sum(grads["dZ"+str(l)],axis=1,keepdims=True)
    
    return grads

def armijo_rule(parameters,grads,X,Y,old_cost,lambd):
    s = 1
    sigma = 0.0001
    beta = 0.5
    m = 0
    mm = X.shape[1]
    old_cost = old_cost*(1/mm)
    L = len(parameters) // 2
    new_parameters = {}
    normgrad = 0
    
    for l in range(L):
        normgrad += np.sum(np.square(grads["dW"+str(l+1)]))
        normgrad += np.sum(np.square(grads["db"+str(l+1)]))
    
    
    for l in range(L):
        new_parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - (beta**m)*s*grads["dW" + str(l+1)]
        new_parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - (beta**m)*s*grads["db" + str(l+1)]
    new_A, A, z = forward_propagation(new_parameters,X)
    new_cost = (1/mm)*compute_loss(new_A,parameters,Y,lambd)
    
    while new_cost > old_cost - sigma*(beta**m)*s*normgrad:
        m += 1
        for l in range(L):
            new_parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - (beta**m)*s*grads["dW" + str(l+1)]
            new_parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - (beta**m)*s*grads["db" + str(l+1)]
        new_A, A, z = forward_propagation(new_parameters,X)
        new_cost = (1/mm)*compute_loss(new_A,parameters,Y,lambd)
    
    return((beta**m)*s)

def update_parameters_with_gd(parameters, grads, learning_rate,X,Y,old_cost,lambd):
    
    L = len(parameters) // 2   # Number of layers
    
    if learning_rate == "armijo":
        learning_rate = armijo_rule(parameters,grads,X,Y,old_cost,lambd)
    else:
        pass
    
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]
    
    return parameters,learning_rate

def update_parameters_with_momentum(parameters, grads, v, learning_rate,X,Y,old_cost,lambd):
    
    L = len(parameters) // 2
    beta = 0.9
    
    for l in range(L):
        v["dW" + str(l+1)] = beta*v["dW" + str(l+1)] + (1-beta)*grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta*v["db" + str(l+1)] + (1-beta)*grads["db" + str(l+1)]
    
    if learning_rate == "armijo":
        learning_rate = armijo_rule(parameters,grads,X,Y,old_cost,lambd)
    else:
        pass   
    
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*v["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] + learning_rate*v["db" + str(l+1)]
        
    return parameters , v , learning_rate

def predict(parameters,X):
    
    last_A, A, z = forward_propagation(parameters,X)
    indexes = np.argmax(last_A,axis=0)
    predict_y = np_utils.to_categorical(indexes, 10)

    return predict_y.T

def accuracy(Y_prediction,Y):
    temp = np.sum(np.abs(Y_prediction - Y),axis=0)
    temp[temp>0] = 1
    return 1 - np.mean(temp)

def convergence(parameters,grads):
    grads_conv = 0
    L = len(parameters) // 2
    
    for l in range(L):
        grads_conv += np.mean(abs(grads["dW" + str(l+1)]))
        grads_conv += np.mean(abs(grads["db" + str(l+1)]))
    
    return grads_conv

def model(X_train_raw ,Y_train_raw, X_test, Y_test, layer_dims, optimizer = "gd", 
          learning_rate = 0.001, num_iter = 5, batch_size = 1024 , decay = False, print_cost = False):
    
    seed = 0
    costs = []
    acc_list_train = []
    acc_list_valid = []
    diff = []
    acc_list_valid.append(0)
    lambd = 0.0005
    learning_rates = []
    convergence_list = []
    
    # Initialize the parameters
    parameters = initialize(layer_dims)
    
    #Initialize the optimizers
    if optimizer == "gd":
        pass
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)

    (X_train,Y_train),(X_train_valid,Y_train_valid) = get_validation_data(X_train_raw, Y_train_raw,seed)
    
    start_time = time.time()
    for i in range(1,num_iter+1):
        
        seed = seed + 1
        mini_batch_X,mini_batch_Y = random_mini_batches(X_train,Y_train,batch_size=batch_size)

        last_A , A , z = forward_propagation(parameters,mini_batch_X)
        cost = (compute_loss(last_A,parameters,mini_batch_Y,lambd))
        ave_cost = cost/batch_size
        costs.append(ave_cost)
        grads = backward_propagation(parameters,A,z,mini_batch_X,mini_batch_Y,lambd)
        convergence_list.append(convergence(parameters,grads))
        

        if optimizer == "gd":
            parameters,learning_r = update_parameters_with_gd(parameters, grads, learning_rate,mini_batch_X,mini_batch_Y,cost,lambd)
            learning_rates.append(learning_r)
        elif optimizer == "momentum":
            parameters,v,learning_r=update_parameters_with_momentum(parameters, grads, v, learning_rate,mini_batch_X,mini_batch_Y,cost,lambd)
            learning_rates.append(learning_r)

        if decay == True and i%50 == 0:
            learning_rate = learning_rate * 0.95
        else:
            pass
   
        # Validation
        if i%5==0:
            Y_prediction_train = predict(parameters,X_train)
            Y_prediction_valid = predict(parameters, X_train_valid)
            accuracy_train = accuracy(Y_prediction_train,Y_train)
            accuracy_valid = accuracy(Y_prediction_valid,Y_train_valid)
            acc_list_train.append(accuracy_train)
            acc_list_valid.append(accuracy_valid)
            diff.append((accuracy_train-accuracy_valid)*100)
        
               
        if print_cost == True and i%100==0:
            print("Cost after %i iteration: %.3f" %(i,ave_cost))
            print("Accuracy Training rate:%.2f" %(accuracy_train*100))
            print("Accuracy Validation rate:%.2f" %(accuracy_valid*100))
            print("The differences is:",(diff[-1]))
            print("--------------------------------------------------")
        
        if i <= 250:
            pass
        else:
            if (diff[-1] > 1 and diff[-2] > 1):
                print("Iteration terminate at:",i)
                break
        
    stop_time = time.time()
    time_taken = stop_time - start_time
    
    Y_prediction_train = predict(parameters,X_train)
    Y_prediction_test = predict(parameters, X_test)
    accuracy_train = accuracy(Y_prediction_train,Y_train)
    accuracy_test = accuracy(Y_prediction_test,Y_test)
    
    '''
    print("The training accuracy is:", accuracy_train*100,"%")
    print("The test accuracy is:",accuracy_test*100,"%")
    print("The differences is:",(accuracy_train-accuracy_test)*100)
    print("The time taken is:",time_taken,"seconds")
    print("-------------------------------------------------------------------")
    '''
    
    dic = { "costs": costs,
            "parameters": parameters,
            "learning_rate":learning_rate,
            "num_iteration": i,
            "time" : time_taken,
            "accuracy_list_train": acc_list_train,
            "accuracy_list_valid": acc_list_valid,
            "accuracy_train": acc_list_train[-1],
            "accuracy_valid": acc_list_valid[-1],
            "accuracy_test" : accuracy_test,
            "learning_rates": learning_rates,
            "convergence" : convergence_list
        }
    
    return dic


X_train,Y_train,X_test,Y_test = get_data("digit")
n_x = X_train.shape[0]
n_y = Y_train.shape[0]


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
        models = model(X_train ,Y_train, X_test, Y_test, layer_dims, optimizer = "gd", 
          learning_rate = i , num_iter = 20 , batch_size = 1024, decay = False, print_cost = False)
        costs_learning_rates.append(model["costs"][-1])
     
    time_stop = time.time()
    total_time = time_stop - time_start
    
    return total_time , learning_rates , costs_learning_rates

def find_architecture(range_of_arc , iterations , data):
    train_list = []
    valid_list = []
    time_list = []

    for i in range_of_arc:
        
        layer_dims = [n_x,i,n_y]
        models = model(X_train ,Y_train, X_test, Y_test, layer_dims, optimizer = "gd", 
          learning_rate = "armijo" , num_iter = iterations, batch_size = 1024 , decay = False, print_cost = False)
        train_list.append(model["accuracy_train"])
        valid_list.append(model["accuracy_valid"])
        time_list.append(model["time"])
        
    return train_list , valid_list , time_list