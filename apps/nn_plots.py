import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 12})

def plots(train_accuracy, test_accuracy, mode, client_id):

    train_plt, = plt.plot(range(1, len(train_accuracy)+1, 1), train_accuracy, label="train")
    test_plt, = plt.plot(range(1, len(test_accuracy)+1, 1), test_accuracy, label="test")

    plt.legend(handles=[train_plt, test_plt], labels=["train", "test"])
    plt.title("Node "+str(client_id)+" - "+mode+" Accuracy")
    plt.xlabel("Training Round")
    plt.ylabel("Accuracy")
    plt.show()

def plots_compare(fair_test_accuracy, local_test_accuracy, unfair_test_accuracy, client_id):

    fair_plt, = plt.plot(range(1, len(fair_test_accuracy)+1, 1), fair_test_accuracy, label="Fair SL")
    local_plt, = plt.plot(range(1, len(local_test_accuracy)+1, 1), local_test_accuracy, label="Local")
    unfair_plt, = plt.plot(range(1, len(unfair_test_accuracy)+1, 1), unfair_test_accuracy, label="Standard SL")
    if(client_id>1):
        plt.axhline(y = np.max(np.asarray(local_test_accuracy)), color = 'grey', linestyle = 'dotted')

    plt.legend(handles=[fair_plt, local_plt, unfair_plt], labels=["Fair SL", "Local", 'Standard SL'], fontsize=12)
    #plt.title("Node "+str(client_id))
    plt.xlabel("Training Round", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.savefig(str(client_id)+"_dataset.eps", dpi=300, format='eps', bbox_inches = 'tight', pad_inches = 0.2)
    plt.show()
    

num_clients = 3

for i in range (num_clients):
    client_id = i+1
    
    client_f_test_accuracy = np.load("./results/fair_node_"+str(client_id)+"_test_acc.npy")
    client_l_test_accuracy = np.load("./results/local_node_"+str(client_id)+"_test_acc.npy")
    client_u_test_accuracy = np.load("./results/unfair_node_"+str(client_id)+"_test_acc.npy")
    client_f_train_accuracy = np.load("./results/fair_node_"+str(client_id)+"_train_acc.npy")
    client_l_train_accuracy = np.load("./results/local_node_"+str(client_id)+"_train_acc.npy")
    client_u_train_accuracy = np.load("./results/unfair_node_"+str(client_id)+"_train_acc.npy")


for i in range (num_clients):
    client_id = i+1
    plots_compare(client_f_test_accuracy, client_l_test_accuracy, client_u_test_accuracy, client_id)

    print("Node_"+str(client_id)+" FairSL acc: ", client_f_test_accuracy[-1])
    print("Node_"+str(client_id)+" unfair acc: ", client_u_test_accuracy[-1])

for i in range (num_clients):
    client_id = i+1   
    plots(client_f_train_accuracy, client_f_test_accuracy, 'FairSwarm', client_id)

for i in range (num_clients):
    client_id = i+1
    plots(client_l_train_accuracy, client_l_test_accuracy, 'Local', client_id)

for i in range (num_clients):
    client_id = i+1
    plots(client_u_train_accuracy, client_u_test_accuracy, 'Unfair', client_id)