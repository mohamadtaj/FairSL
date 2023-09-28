from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
import math
from copy import deepcopy
from tensorflow.keras import regularizers
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from utils import *
from models import *



class Node:

    def __init__(self, env, x_train, y_train, nodes_sizes, id, estimators):

        self.id = id
        self.env = env
        self.peers = None
        self.connections = {}
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = None
        self.y_test = None        
        self.nn_model = None
        self.nn_params = None
        self.nn_batch_loss = []
        self.nn_batch_accuracy = []
        self.nn_train_loss = []
        self.nn_test_loss = []
        self.nn_train_accuracy = []
        self.nn_test_accuracy = []          
        self.more_batches = True
        self.train_batches = None
        self.test_batches = None
        self.sync = None
        self.num_batches = None
        self.x_train_list = None
        self.y_train_list = None      
        self.active = True      
        self.input_size = None
        self.output_size = len(np.unique(self.y_train))     
        self.next_coordinator = None
        
        self.x_train_sections = None
        self.y_train_sections = None        
        self.network_params = []
        self.nodes_sizes = nodes_sizes
        self.nodes_sizes_list = list(self.nodes_sizes.values())
        self.rf_model = None
        self.rf_final = None
        self.num_estimators = estimators
        self.contributions = None
        self.unique_classes = np.unique(self.y_train)
        self.num_classes = len(self.unique_classes)
        self.fold_generator = None
        self.train_folds_idx = []
        self.test_folds_idx = []        
        self.cv_results_acc = []
        self.cv_results_f1 = []
        self.cv_results_mcc = []
        self.cv_score_acc = None
        self.cv_score_f1 = None
        self.cv_score_mcc = None       
        self.accuracy = None   
        self.f1 = None
        self.mcc_ = None
    
    # connect the node to a given peer    
    def connect(self, peer):
        conn = Connection(self, peer)
        conn.active = True
        self.connections[peer] = conn
        if not peer.is_connected(self):
            peer.connect(self)

    # connect the node to all the other nodes
    def connect_all(self):
        for peer in self.peers:
            self.connect(peer)

    # set the node's peers            
    def set_peers(self):
        self.peers = self.env.nodes.copy()
        self.peers.remove(self)

    # check if there is a conenction between the node and a given peer
    def is_connected(self, peer):
        return peer in self.connections

    def send(self, receiver, msg, broadcast):
        conn = self.connections[receiver]
        conn.deliver(msg, broadcast)

    def receive(self, msg, broadcast):
        if (broadcast):
            self.update_nn_params(msg)
        else:    
            self.network_params.append(msg)
           

    # Share random forest estimators based on the relative data sizes
    def share_rf_params_fair(self):
        num_samples = len(self.x_train)
        for peer in self.peers:
            fraction = self.nodes_sizes[peer.id] / num_samples
            idx = math.ceil(self.num_estimators * fraction)    
            
            if (num_samples > len(peer.x_train)):
                idx = math.ceil(idx * fraction)
            
            if (idx > self.num_estimators):
                idx = self.num_estimators   

            rf_params = self.rf_model.estimators_[:idx]
            
            broadcast = False
            self.send(peer, rf_params, broadcast)

    # Share all rf estimators
    def share_rf_params_all(self):
        broadcast = False
        for peer in self.peers:
            self.send(peer, self.rf_model.estimators_, broadcast)

    # Update the random forest based on the received estimators from others
    def update_rf(self):
        rf_final = deepcopy(self.rf_model)
        
        for params in self.network_params:
            rf_final.estimators_ += params
        rf_final.n_estimators_ = len(rf_final.estimators_)
        self.network_params = []
        self.rf_final = rf_final   

    # Initialize RF model
    def define_rf_model(self):
        self.rf_model = RandomForestClassifier(n_estimators = self.num_estimators)

    # Initialize DeepL model
    def define_nn_model(self, dataset, OPTIMIZER, LOSS, input_size, output_size):
        model = nn_model(dataset, input_size, output_size)
        model.compile(optimizer = OPTIMIZER, loss = LOSS, metrics = ['accuracy'])
        self.nn_params = model.get_weights()
        self.nn_model = model
        
    # split the dataset into n parts of different sizes - n: 1 to number of nodes    
    def split_dataset(self, index):
                
        x = np.array_split(self.x_train, self.nodes_sizes_list[:index])
        self.x_train_list = x[:-1]
        y = np.array_split(self.y_train, self.nodes_sizes_list[:index])
        self.y_train_list = y[:-1]

    def update_nn_params(self, params):
        self.nn_params = params

    def total_batches(self, BATCH_SIZE):
        return len (mini_batches(self.x_train, self.y_train, BATCH_SIZE))
        
    def train_rf (self, cv_round, aug):
        
        train_idx = self.train_folds_idx[cv_round]
            
        x_train = self.x_train[train_idx]
        y_train = self.y_train[train_idx]
        if (aug == True):
            x_train, y_train = self.smote(x_train, y_train)

        self.rf_model.fit(x_train, y_train)

    def train_nn(self, BATCH_SIZE, mode, *section):
        
        if (mode == 'fair'):
            sect = section[0]
            self.train_batches = mini_batches(self.x_train_list[sect], self.y_train_list[sect], BATCH_SIZE)
            
        elif(mode == 'unfair' or mode == 'local'):
            self.train_batches = mini_batches(self.x_train, self.y_train, BATCH_SIZE)
            
        BATCH_NUM = len(self.train_batches)
        self.nn_model.set_weights(self.nn_params)
        batch_loss = []
        batch_accuracy = []

        for batch_iter in range(BATCH_NUM):
        
            x, y = self.train_batches[batch_iter]
            loss, accuracy = self.nn_model.train_on_batch(x,y)
            batch_loss = np.append(batch_loss, loss)
            batch_accuracy = np.append(batch_accuracy, accuracy)

        self.nn_train_loss = np.append(self.nn_train_loss, np.mean(batch_loss))
        self.nn_train_accuracy = np.append(self.nn_train_accuracy, np.mean(batch_accuracy)) 
        
        weights = self.nn_model.get_weights()

        self.update_nn_params(weights)
        
    def evaluate_nn (self, BATCH_SIZE):
    
        self.test_batches = mini_batches(self.x_test, self.y_test, BATCH_SIZE)
        BATCH_NUM = len(self.test_batches)
        self.nn_model.set_weights(self.nn_params)  
        batch_loss = []
        batch_accuracy = []
    
        for batch_iter in range(BATCH_NUM):
        
            x, y = self.test_batches[batch_iter]
            loss, accuracy = self.nn_model.test_on_batch(x,y)
            batch_loss = np.append(batch_loss, loss)
            batch_accuracy = np.append(batch_accuracy, accuracy)

        self.nn_test_loss = np.append(self.nn_test_loss, np.mean(batch_loss))
        self.nn_test_accuracy = np.append(self.nn_test_accuracy, np.mean(batch_accuracy))  

    
    # Share neural network parameters
    def share_nn_params(self):
        broadcast = False # parameters are sent from nodes to the coordinator node
        if(self.next_coordinator != self):
            msg = (self.id, self.nn_params)
            self.send(self.next_coordinator, msg, broadcast)
            
    
    # Take the average of the received parameters from nodes in a fair manner
    def take_avg_params_fair(self):
        self_params = (self.id, self.nn_params)
        self.network_params.append(self_params)
        params = np.array([x[1] for x in self.network_params])
                
        avg = np.mean(params, axis=0)
        self.network_params = []
        self.update_nn_params (avg)

    # Take the average of the received parameters from nodes based on FedAvg (unfair)    
    def take_avg_params_unfair(self):   
        self_params = (self.id, self.nn_params)
        self.network_params.append(self_params)
        
        ids = [x[0] for x in self.network_params]
        params = np.array([x[1] for x in self.network_params])
        sizes = np.array([self.nodes_sizes[id] for id in ids])
        
        avg = np.dot(sizes, params)/np.sum(sizes)
        self.network_params = []
        self.update_nn_params (avg)      

    # Send the new parameters to all the nodes in the network
    def broadcast_nn_params(self): 
        broadcast = True # Parameters are send from the coordinator to all the nodes
        for peer in self.peers:
            self.send(peer, self.nn_params, broadcast)
   
    def predict_rf(self, cv_round):
        test_idx = self.test_folds_idx[cv_round]
            
        x_test = self.x_train[test_idx]
        y_test = self.y_train[test_idx]

        y_pred = self.rf_final.predict(x_test)
        
        self.accuracy = metrics.accuracy_score(y_test, y_pred)
        self.f1 = metrics.f1_score(y_test, y_pred, average='macro')
        self.mcc = metrics.matthews_corrcoef(y_test, y_pred)
        self.cv_results_acc.append(self.accuracy)
        self.cv_results_f1.append(self.f1)            
        self.cv_results_mcc.append(self.mcc)         

    # Create folds for Kfold cross-validation for random forest
    def create_folds(self, cv_splits):
        kfold = StratifiedKFold(n_splits = cv_splits, shuffle=True, random_state=0)
        
        self.fold_generator = kfold.split(self.x_train, self.y_train)
        for train_ix, test_ix in self.fold_generator:
            train_index = train_ix
            test_index  = test_ix
            self.train_folds_idx.append(train_index)
            self.test_folds_idx.append(test_index)        
        
    def cross_val_score(self):
        self.cv_score_acc = np.mean(self.cv_results_acc)
        self.cv_score_f1 = np.mean(self.cv_results_f1)
        self.cv_score_mcc = np.mean(self.cv_results_mcc)

    # Use oversampling for unbalanced datasets                        
    def smote (self, X, y):
        oversample = SMOTE()
        X, y = oversample.fit_resample(X, y)
        return X, y        
        
    def print_info(self):
        print()
        print(f'node {self.id}:')
        print(f'len x_train: {len(self.x_train)}')
        print(f'num classes: {self.num_classes}')
        print(f'unique classes: {self.unique_classes}')
        for c in self.unique_classes:
            print(f'class {c} size: {np.count_nonzero(self.y_train == c)}')
        print()

# managing the connection between two peersfor sending and receiving parameters 
class Connection:
    def __init__(self, sender, receiver):
        self.sender = sender
        self.receiver = receiver
        
    def deliver(self, msg, broadcast):
        self.receiver.receive(msg, broadcast)

# simulation environment 
class Env:
    def __init__(self, X, y, num_nodes, fractions, n_estimators=None):
        self.num_nodes = num_nodes
        self.nodes = None
        self.X = X
        self.y = y
        self.fractions = fractions
        self.n_estimators = n_estimators
        self.next_coordinator = None
                
    def create_nodes(self, seed):

        m = self.X.shape[0]
        np.random.seed(seed)
        permutation = np.random.permutation(m)
        shuffled_X = self.X[permutation]
        shuffled_Y = self.y[permutation]

        samples = split(m, self.fractions)
        arr = np.cumsum(samples)
        
        estimators = split (self.n_estimators, self.fractions)

        X_node = np.array_split(shuffled_X, arr)
        Y_node = np.array_split(shuffled_Y, arr)
        
        """Nodes sizes as a dictionary"""
        nodes_sizes_dict = {}
        for i in range(self.num_nodes):
            nodes_sizes_dict[i+1] = samples[i]
        """Nodes sizes as a dictionary"""

        nodes = [Node(self, X_node[i], Y_node[i], nodes_sizes_dict, i+1, estimators[i]) for i in range (self.num_nodes)]

        self.nodes = nodes
        self.nodes_sizes = nodes_sizes_dict
    
    # initialize peers at the network    
    def initialize_peers(self):
        for node in self.nodes:
            node.set_peers()

    # make a connection among all the nodes            
    def connect_net(self):
        for node in self.nodes:
            node.connect_all() 

    # Randomly set a coordinator for the next training round
    def set_coordinator(self):
        active_nodes = [node for node in self.nodes if node.active==True]

        lucky = np.random.choice(active_nodes)
        self.next_coordinator = lucky
        for node in self.nodes:
            node.next_coordinator = lucky           