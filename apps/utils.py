from data_loader import *
import math
from keras import backend as K

def scale (x_train, x_test):

    x_train = x_train.astype("float32")/255.
    x_test = x_test.astype("float32")/255.

    return x_train, x_test

def load_data_rf(data):

    X, y = load_dataset_rf(data) 
    
    return X, y

# def load_data_nn(data, data_type):

    # x_train, y_train, x_test, y_test = load_dataset_nn(data)
    # if (data_type == 'img'):
        # x_train, x_test = scale (x_train, x_test)

    # return x_train, y_train, x_test, y_test  

def load_data_nn(data, data_type):

    x_train, y_train, x_test, y_test = load_dataset_nn(data)
    if (data_type == 'img'):
        if(not data=='cifar100'):
            x_train, x_test = scale (x_train, x_test)
        else:
            print('Already preprocessed')

    return x_train, y_train, x_test, y_test  
    
def split(n, fractions):
    
    result = []
    for fraction in fractions[:-1]:
        result.append(round(fraction * n))
    result.append(n - np.sum(result))
    
    return result
    
def mini_batches(X, Y, mini_batch_size):
    
    m = X.shape[0]
    mini_batches = []

    permutation = np.random.permutation(m)
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation]

    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[k*mini_batch_size : (k+1)*mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m % mini_batch_size != 0:

        mini_batch_X = shuffled_X[(k+1)*mini_batch_size : m]
        mini_batch_Y = shuffled_Y[(k+1)*mini_batch_size : m]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
    
def reduce_size (X, y, train_size):

    m = X.shape[0]

    np.random.seed(0)
    permutation = np.random.permutation(m)

    shuffled_X = X[permutation,:]
    shuffled_Y = y[permutation]

    reduced_X = shuffled_X[:train_size]
    reduced_Y = shuffled_Y[:train_size]

    return reduced_X, reduced_Y    

def freeze_layers (node, dataset, cycle, num_nodes):
    if (dataset == 'cifar10'):
        freeze_layers_cifar10(node, cycle)
    elif(dataset == 'reuters'):
        freeze_layers_reuters(node, cycle)
    elif(dataset == 'cifar100'):
        freeze_layers_cifar100(node, cycle, num_nodes)
    elif(dataset == 'mnist'):
        freeze_layers_mnist(node, cycle, num_nodes)
        
def freeze_layers_cifar10(node, cycle):
    if (cycle == 2):

        K.set_value(node.nn_model.optimizer.learning_rate, 0.0005)
        
        node.nn_model.layers[0].trainable = False
        node.nn_model.layers[4].trainable = False    

    elif (cycle == 3):

        K.set_value(node.nn_model.optimizer.learning_rate, 0.0001)
        
        node.nn_model.layers[0].trainable = False
        node.nn_model.layers[4].trainable = False 
        node.nn_model.layers[8].trainable = False 

def freeze_layers_mnist(node, cycle):
    if (num_nodes==3):
        if (cycle == 2):
            K.set_value(node.nn_model.optimizer.learning_rate, 0.0008)
       
            for layer in node.nn_model.layers[:5]: 
                layer.trainable = False 
                
        elif (cycle == 3):
            K.set_value(node.nn_model.optimizer.learning_rate, 0.0001)

            for layer in node.nn_model.layers[:12]: 
                layer.trainable = False 
    
    elif(num_nodes==4):
        if (cycle == 2):
            K.set_value(node.nn_model.optimizer.learning_rate, 0.0001)
   
            for layer in node.nn_model.layers[:12]: 
                layer.trainable = False 
                
        elif (cycle == 3):
            K.set_value(node.nn_model.optimizer.learning_rate, 0.001)

            for layer in node.nn_model.layers: 
                layer.trainable = True 
                
            for layer in node.nn_model.layers[:5]: 
                layer.trainable = False 
        
        elif (cycle == 4):  
            K.set_value(node.nn_model.optimizer.learning_rate, 0.0001)

            for layer in node.nn_model.layers[:12]: 
                layer.trainable = False         
        
def freeze_layers_cifar100(node, cycle):
    if (num_nodes==3):
        if (cycle == 2):

            K.set_value(node.nn_model.optimizer.learning_rate, 0.003)

        elif (cycle == 3):

            K.set_value(node.nn_model.optimizer.learning_rate, 0.0005)
    
    elif(num_nodes==4):
        if (cycle == 2):

            K.set_value(node.nn_model.optimizer.learning_rate, 0.0001)

        elif (cycle == 3):

            K.set_value(node.nn_model.optimizer.learning_rate, 0.001)
            
        elif (cycle == 4):

            K.set_value(node.nn_model.optimizer.learning_rate, 0.0001)    
        
def freeze_layers_reuters(node, cycle):
    if (cycle == 2):

        K.set_value(node.nn_model.optimizer.learning_rate, 0.0008) 

    elif (cycle == 3):

        K.set_value(node.nn_model.optimizer.learning_rate, 0.0003)
        
def cross_val_scores(env):
    for node in env.nodes:
        node.cross_val_score()
 
def generate_folds(env, cv_splits):
    for node in env.nodes:
        node.create_folds(cv_splits)

def export_results_rf (env, mode, nodes_results_acc, nodes_results_f1, nodes_results_mcc):

    if not os.path.exists("results"):
        os.makedirs("results")
        
    for node in env.nodes: 
        np.save('results/'+str(mode)+'_node_'+str(node.id)+'_acc.npy', np.dot(100, nodes_results_acc[node.id]))
        np.save('results/'+str(mode)+'_node_'+str(node.id)+'_f1.npy', nodes_results_f1[node.id])
        np.save('results/'+str(mode)+'_node_'+str(node.id)+'_mcc.npy', nodes_results_mcc[node.id])

def export_results_nn (env, mode):

    if not os.path.exists("results"):
        os.makedirs("results")
        
    for node in env.nodes: 
        np.save('results/'+str(mode)+'_node_'+str(node.id)+'_train_loss.npy', node.nn_train_loss)
        np.save('results/'+str(mode)+'_node_'+str(node.id)+'_train_acc.npy', np.dot(100, node.nn_train_accuracy))
        np.save('results/'+str(mode)+'_node_'+str(node.id)+'_test_loss.npy', node.nn_test_loss)
        np.save('results/'+str(mode)+'_node_'+str(node.id)+'_test_acc.npy', np.dot(100, node.nn_test_accuracy))    