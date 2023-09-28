import time
from utils import *


def train_rf_network (env, cv_round, aug, mode):
    for node in env.nodes:
        node.define_rf_model()
        node.train_rf (cv_round, aug)
        
        if(mode == 'unfair'):
            node.share_rf_params_all()
        elif(mode == 'fair'):
            node.share_rf_params_fair()
    
    for node in env.nodes:
        node.update_rf()
        
def evaluate_rf_network(env, cv_round):
    for node in env.nodes:
        node.predict_rf(cv_round)
        
        
def train_nn_network_fair (env, mode, BATCH_SIZE, section, OPTIMIZER, LOSS):
    assert(mode == 'fair')
    for node in env.nodes:
        if(node.active):
            node.train_nn(BATCH_SIZE, mode, section)
            node.share_nn_params()
    
    coordinator = env.next_coordinator   
    coordinator.take_avg_params_fair()     
    coordinator.broadcast_nn_params()
    
def train_nn_network_unfair (env, mode, BATCH_SIZE, OPTIMIZER, LOSS):
    assert(mode == 'unfair')
    for node in env.nodes:
        if(node.active):
            node.train_nn(BATCH_SIZE, mode)
            node.share_nn_params()
    
    coordinator = env.next_coordinator   
    coordinator.take_avg_params_unfair()    
    coordinator.broadcast_nn_params()


def train_nn_network_local (env, mode, BATCH_SIZE, OPTIMIZER, LOSS):
    assert(mode == 'local')
    for node in env.nodes:
        if(node.active):
            node.train_nn(BATCH_SIZE, mode)    
        
def evaluate_nn_network (env, BATCH_SIZE):
    for node in env.nodes:
        if(node.active):
            node.evaluate_nn(BATCH_SIZE)
        
        
def run_rf_sim(env, aug, cv_splits, mode):

    print()
    if (mode == 'unfair'):
        print("Typical (Unfair) Swarm RF Model Training... ")
    if (mode == 'fair'):
        print("Fair Swarm RF Model Training... ")
    if (mode == 'local'):
        print("Local RF Model Training... ")

    start_time = time.time()
    
    generate_folds(env, cv_splits)
    
    for cv_round in range(cv_splits):  
        train_rf_network (env, cv_round, aug, mode)
        evaluate_rf_network(env, cv_round)
        
    cross_val_scores(env)

    print(); print("Execution Time %s seconds: " % (time.time() - start_time)) 


def run_nn_sim_fair (env, mode, dataset, BATCH_SIZE, EPOCHS, OPTIMIZER, LOSS, input_size, output_size):

    assert (mode == 'fair')
    print()
    print("---------------------------------------------------------------------")
    print('Fair Swarm Neural Network Training...')
    print()
    
    start_time = time.time()

    idx = 1
    for node in env.nodes:
        node.split_dataset(idx)
        idx+=1
        
    for node in env.nodes:
        print("Node {} num of sections: {}".format(node.id, len(node.x_train_list)))
        print("sections lengths:")
        for i in range(len(node.x_train_list)):
            print(len(node.x_train_list[i]))
        print("________________________________________________")

    for node in env.nodes:
        node.define_nn_model(dataset, OPTIMIZER, LOSS, input_size, output_size)
    

    section=0    
    cycle = 1
    
    for train_cycle in range (env.num_nodes):
    
        print()
        print("---------------------------------------------------------------------------------")
        print(f'Cycle {cycle} Training ... ')
        print("Nodes Participating: ")
        for node in env.nodes:
            if(node.active):    
                freeze_layers(node, dataset, cycle)                  

        for epoch_num in range(EPOCHS):
                 
            train_nn_network_fair (env, mode, BATCH_SIZE, section, OPTIMIZER, LOSS)
            evaluate_nn_network (env, BATCH_SIZE)

            print("EPOCH {} FINISHED".format(epoch_num + 1))
            print()
            for node in env.nodes:
                print(f'----------------------------Node {node.id} Train Loss = {node.nn_train_loss[-1]}')        
                print(f'----------------------------Node {node.id} Train Accuracy = {node.nn_train_accuracy[-1]}')
                print(f'----------------------------Node {node.id} Test Loss = {node.nn_test_loss[-1]}')
                print(f'----------------------------Node {node.id} Test Accuracy = {node.nn_test_accuracy[-1]}')
                print()
                print(f'--------------------------------------------Coordinator Id: {env.next_coordinator.id}')
                print()

            env.set_coordinator()
            
        env.nodes[section].active = False
        section+=1
        cycle+=1

    print(); print("Execution Time %s seconds: " % (time.time() - start_time))

def run_nn_sim_unfair (env, mode, dataset, BATCH_SIZE, EPOCHS, OPTIMIZER, LOSS, input_size, output_size):

    assert (mode == 'unfair')
    print()
    print("---------------------------------------------------------------------")
    print('Typical Swarm (Unfair) Neural Network Training...')
    print()
    
    start_time = time.time()
    
    for node in env.nodes:
        node.define_nn_model(dataset, OPTIMIZER, LOSS, input_size, output_size)
        
    for epoch_num in range(EPOCHS):
    
        train_nn_network_unfair (env, mode, BATCH_SIZE, OPTIMIZER, LOSS)
        evaluate_nn_network (env, BATCH_SIZE)
        
        print("EPOCH {} FINISHED".format(epoch_num + 1))
        print()
        for node in env.nodes:
            print(f'----------------------------Node {node.id} Train Loss = {node.nn_train_loss[-1]}')        
            print(f'----------------------------Node {node.id} Train Accuracy = {node.nn_train_accuracy[-1]}')
            print(f'----------------------------Node {node.id} Test Loss = {node.nn_test_loss[-1]}')
            print(f'----------------------------Node {node.id} Test Accuracy = {node.nn_test_accuracy[-1]}')
            print()

    print(); print("Execution Time %s seconds: " % (time.time() - start_time))
   
def run_nn_sim_local (env, mode, dataset, BATCH_SIZE, EPOCHS, OPTIMIZER, LOSS, input_size, output_size):

    assert (mode == 'local')
    print()
    print("---------------------------------------------------------------------")
    print('Local Neural Network Training...')
    print()
    
    start_time = time.time()

    for node in env.nodes:
        node.define_nn_model(dataset, OPTIMIZER, LOSS, input_size, output_size)
    
    
    for epoch_num in range(EPOCHS):
    
        train_nn_network_local (env, mode, BATCH_SIZE, OPTIMIZER, LOSS)
        evaluate_nn_network (env, BATCH_SIZE)

        print("EPOCH {} FINISHED".format(epoch_num + 1))
        print()
        for node in env.nodes:

            print(f'----------------------------Node {node.id} Train Loss = {node.nn_train_loss[-1]}')        
            print(f'----------------------------Node {node.id} Train Accuracy = {node.nn_train_accuracy[-1]}')
            print(f'----------------------------Node {node.id} Test Loss = {node.nn_test_loss[-1]}')
            print(f'----------------------------Node {node.id} Test Accuracy = {node.nn_test_accuracy[-1]}')
            print()


    print(); print("Execution Time %s seconds: " % (time.time() - start_time))            