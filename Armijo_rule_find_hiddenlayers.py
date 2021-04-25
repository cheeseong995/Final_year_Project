# This is used to find the number of nodes in the hidden layers

import matplotlib.pyplot as plt
import master_code_final_v1 as mcf1

X_train,Y_train,X_test,Y_test = mcf1.get_data("digit")
n_x = X_train.shape[0]
n_y = Y_train.shape[0]

def find_architecture(range_of_arc , iterations , data):
    train_list = []
    valid_list = []
    time_list = []

    for i in range_of_arc:
        
        layer_dims = [n_x,i,n_y]
        model = mcf1.model(X_train ,Y_train, X_test, Y_test, layer_dims, optimizer = "gd", 
          learning_rate = "armijo" , num_iter = iterations, batch_size = 1024 , decay = False, print_cost = False)
        train_list.append(model["accuracy_train"])
        valid_list.append(model["accuracy_valid"])
        time_list.append(model["time"])
        
    return train_list , valid_list , time_list

# First search for architecture for digit data set
# Geometric pyramid rule states that it is around 88. Range from [70:110]

range_of_arc = list(range(40,141,10))
train_list , valid_list , time_list = find_architecture(range_of_arc , 100 , "digit")

plt.plot(train_list , color = "red" , label = "Train")
plt.plot(valid_list , color = "green" , label = "Valid")
x = list(range(len(range_of_arc)))
plt.xticks(x,range_of_arc)
plt.ylabel('Accuracy')
plt.title("1st Grid-Search (MNIST Digit)")
plt.xlabel('Number of Nodes')
plt.legend(loc="lower left",shadow = True)
ax = plt.gca()
ax.grid(True)
plt.show()

plt.plot(time_list)
x = list(range(len(range_of_arc)))
plt.xticks(x,range_of_arc)
plt.ylabel('Time taken')
plt.title("1st Grid-Search (MNIST Digit)")
plt.xlabel('Number of Nodes')
ax = plt.gca()
ax.grid(True)
plt.show()

range_of_arc = list(range(60,101,10))
train_list , valid_list , time_list = find_architecture(range_of_arc , 250 , "digit")

plt.plot(train_list , color = "red" , label = "Train")
plt.plot(valid_list , color = "green" , label = "Valid")
x = list(range(len(range_of_arc)))
plt.xticks(x,range_of_arc)
plt.ylabel('Accuracy')
plt.title("2nd Grid-Search (MNIST Digit)")
plt.xlabel('Number of Nodes')
plt.legend(loc="center right",shadow = True)
ax = plt.gca()
ax.grid(True)
plt.show()

plt.plot(time_list)
x = list(range(len(range_of_arc)))
plt.xticks(x,range_of_arc)
plt.ylabel('Time taken')
plt.title("2nd Grid-Search (MNIST Digit)")
plt.xlabel('Number of Nodes')
ax = plt.gca()
ax.grid(True)
plt.show()

#--------------------------------------------------------------------------------------------

X_train,Y_train,X_test,Y_test = mcf1.get_data("fashion")
n_x = X_train.shape[0]
n_y = Y_train.shape[0]

range_of_arc = list(range(200,300,10))
train_list , valid_list , time_list = find_architecture(range_of_arc , 50 , "fashion")

plt.plot(train_list , color = "red" , label = "Train")
plt.plot(valid_list , color = "green" , label = "Valid")
x = list(range(len(range_of_arc)))
plt.xticks(x,range_of_arc)
plt.ylabel('Accuracy')
plt.title("1st Grid-Search (MNIST Fashion)")
plt.xlabel('Number of Nodes')
plt.legend(loc="upper right",shadow = True)
ax = plt.gca()
ax.grid(True)
plt.show()

plt.plot(time_list)
x = list(range(len(range_of_arc)))
plt.xticks(x,range_of_arc)
plt.ylabel('Time taken')
plt.title("1st Grid-Search (MNIST Fashion)")
plt.xlabel('Number of Nodes')
ax = plt.gca()
ax.grid(True)
plt.show()


range_of_arc = list(range(240,281,10))
train_list , valid_list , time_list = find_architecture(range_of_arc , 100 , "fashion")

plt.plot(train_list , color = "red" , label = "Train")
plt.plot(valid_list , color = "green" , label = "Valid")
x = list(range(len(range_of_arc)))
plt.xticks(x,range_of_arc)
plt.ylabel('Accuracy')
plt.title("2nd Grid-Search (MNIST Fashion)")
plt.xlabel('Number of Nodes')
plt.legend(loc="center right",shadow = True)
ax = plt.gca()
ax.grid(True)
plt.show()

plt.plot(time_list)
x = list(range(len(range_of_arc)))
plt.xticks(x,range_of_arc)
plt.ylabel('Time taken')
plt.title("2nd Grid-Search (MNIST Fashion)")
plt.xlabel('Number of Nodes')
ax = plt.gca()
ax.grid(True)
plt.show()

#Final Layers: 80 and 260
