import argparse
from main import *
import tensorflow as tf
parser = argparse.ArgumentParser()

# Parse command line arguments
parser.add_argument("--dataset", "--dataset", type=str, help="Name of the dataset")
parser.add_argument("--framework", "--framework", type=str, help="The machine learning framework (Neural Networks: nn, Random Forest: rf)")
parser.add_argument("--mode", "--mode", type=str, help="mode of training (Fair SL: fair, Typical SL: unfair, Local: local)")
args = vars(parser.parse_args())
 
# Set up parameters
data = args["dataset"]
framework = args["framework"]
mode = args["mode"]
seed = 0
num_nodes = 3
node_fractions = [0.1, 0.3, 0.6] 

if (framework == 'rf'):
    iterations = 100
    
# Neural networks parameters
BATCH_SIZE = 128

if (data == 'cifar10'):
    data_type = 'img'
    OPTIMIZER = tf.keras.optimizers.Adam()
    LOSS = 'sparse_categorical_crossentropy'
    iterations = 100
    
elif (data == 'reuters'):   
    data_type = 'txt'
    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.003)
    LOSS = 'sparse_categorical_crossentropy'
    iterations = 200

# Random forest parameters

if (data == 'android'):
    n_estimators = 500
    augmentation = False
    cv_splits = 10

elif (data == 'breast_cancer'):
    n_estimators = 100
    augmentation = False
    cv_splits = 5

elif (data == 'heart_failure'):
    n_estimators = 500
    augmentation = False
    cv_splits = 10
    
elif (data == 'maternal_health'):
    n_estimators = 500
    augmentation = False
    cv_splits = 10

elif (data == 'auction'):
    n_estimators = 1000
    augmentation = True
    cv_splits = 10

elif (data == 'student'):
    n_estimators = 300
    augmentation = False
    cv_splits = 10

    
if (framework == 'rf'):
    main_rf(data, num_nodes, node_fractions, n_estimators, augmentation, cv_splits, iterations, seed, mode)

elif(framework == 'nn'):
    main_nn (data, num_nodes, node_fractions, iterations, seed, mode, BATCH_SIZE, OPTIMIZER, LOSS, data_type)