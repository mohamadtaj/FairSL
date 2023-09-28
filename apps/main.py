import warnings
warnings.filterwarnings("ignore")

import numpy as np
import time
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from peer import *
from utils import *
from sim import *

config = ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = InteractiveSession(config=config)

def main_rf(data, num_nodes, node_fractions, n_estimators, augmentation, cv_splits, iterations, seed, mode):

    DATASET_NAME = data
    X, y = load_data_rf(DATASET_NAME)

    nodes_results_acc = {}
    nodes_results_f1 = {}
    nodes_results_mcc = {}

    for i in range (1, num_nodes+1):
        nodes_results_acc[i] = []       
        nodes_results_f1[i] = []
        nodes_results_mcc[i] = []
        
          
    for i in range(iterations):
   
        print()
        print('___________________________________________________________________________')
        print(f'Round {i+1} ... ')   
        
        env = Env(X, y, num_nodes, node_fractions, n_estimators)
        env.create_nodes(seed)
               
        if (mode == 'unfair' or mode == 'fair'):
            env.initialize_peers()
            env.connect_net()    
            for node in env.nodes:        
                node.connect_all()
            env.set_coordinator()
         
        run_rf_sim(env, augmentation, cv_splits, mode)

        for node in env.nodes:
        
            nodes_results_acc[node.id].append(node.cv_score_acc)
            nodes_results_f1[node.id].append(node.cv_score_f1)
            nodes_results_mcc[node.id].append(node.cv_score_mcc)
        
        seed += 1 

    print('-----------------------------------------------------------------');print()
    print('FINAL RESULTS:');print()

    print('MCC Scores:')    
    for i in range (1, num_nodes+1):
        print(f'node {i}:')
        print(f'-------------Mean: {np.mean(nodes_results_mcc[i]):.2f}')   
        print(f'-------------std: {np.std(nodes_results_mcc[i]):.2f}') 

    export_results_rf (env, mode, nodes_results_acc, nodes_results_f1, nodes_results_mcc)
        
        
        
def main_nn (data, num_nodes, node_fractions, EPOCHS, seed, mode, BATCH_SIZE, OPTIMIZER, LOSS, data_type):
    
    DATASET_NAME = data
    x_train, y_train, x_test, y_test = load_data_nn(DATASET_NAME, data_type)         

    unique_classes = len(np.unique(y_train)) # number of classes for classification

    # determining the number of layers at the output based on the number of classes
    if(unique_classes == 2):
        output_size = 1
    elif(unique_classes > 2):
        output_size = unique_classes 
        
    if(data_type == 'img'):
        input_size = x_train.shape[1:]
            
    elif(data_type == 'txt'):
        input_size = None
    
    estimators = 0
    env = Env(x_train, y_train, num_nodes, node_fractions, estimators)
    env.create_nodes(seed)
    
    for node in env.nodes:
        node.x_test = x_test
        node.y_test = y_test
    
    if (mode == 'unfair' or mode == 'fair'):
        env.initialize_peers()
        env.connect_net()    
        for node in env.nodes:        
            node.connect_all()
        env.set_coordinator() 
          
        
    if(mode == 'fair'):
        run_nn_sim_fair (env, mode, DATASET_NAME, BATCH_SIZE, EPOCHS, OPTIMIZER, LOSS, input_size, output_size)  
    if(mode == 'unfair'):
        run_nn_sim_unfair (env, mode, DATASET_NAME, BATCH_SIZE, EPOCHS, OPTIMIZER, LOSS, input_size, output_size)  
    if(mode == 'local'):
        run_nn_sim_local (env, mode, DATASET_NAME, BATCH_SIZE, EPOCHS, OPTIMIZER, LOSS, input_size, output_size) 

    export_results_nn (env, mode)