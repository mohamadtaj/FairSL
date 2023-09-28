import pandas as pd
import numpy as np
import os
import cv2
from tensorflow.keras.datasets import cifar10, reuters
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
#from functools import partial  
    
"""load datasets for deep learning"""   
def load_dataset_nn(data):

    if (data =="cifar" or data =="cifar10" or data =='CIFAR' or data =='CIFAR10'):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = y_train.flatten()
        y_test = y_test.flatten()      

        
    elif (data =='reuters'):
        (x_train, y_train), (x_test, y_test) = reuters.load_data(
                                            num_words=10000,
                                            skip_top=20,
                                            maxlen=500,
                                            test_split=0.2,
                                            seed=0)       
        x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=500) 
        x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=500)        
      
    return x_train, y_train, x_test, y_test

"""load datasets for random forest classification"""      
def load_dataset_rf(data):   
    if (data == 'breast_cancer'):

        path = './datasets/breast_cancer'
        data = pd.read_csv(os.path.join(path,'breast_cancer.csv'), encoding = 'utf-8')
      
        x = data.iloc[:,:-1].copy()
        y = data['Classification'].copy()
        
        x = x.to_numpy()
        y = y.to_numpy()
        
        m = x.shape[0]
        np.random.seed(0)
        permutation = np.random.permutation(m)
        x = x[permutation]
        y = y[permutation] 
        
        print('Train data shape: ',x.shape)
        print('Train y shape: ',y.shape) 

    elif (data == 'maternal_health'):

        path = './datasets/maternal_health_risk'
        data = pd.read_csv(os.path.join(path,'Maternal Health Risk Data Set.csv'), encoding = 'utf-8')

        x = data.iloc[:,:-1].copy()
        y = data['RiskLevel'].copy()  
        
        x = x.to_numpy()
        y = y.to_numpy()
        
        le = LabelEncoder()
        y = le.fit_transform(y) 
 
        m = x.shape[0]
        np.random.seed(0)
        permutation = np.random.permutation(m)
        x = x[permutation]
        y = y[permutation] 
        
        print('Train data shape: ',x.shape)     
        
    elif (data == 'student'):

        path = './datasets/student_dropout'
        data = pd.read_csv(os.path.join(path,'student.csv'), delimiter = ';')

        x = data.iloc[:,:-1].copy()
        y = data['Target'].copy()  
        
        y.replace({'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}, inplace=True)
        
        x = x.to_numpy()
        y = y.to_numpy()
 
        m = x.shape[0]
        np.random.seed(0)
        permutation = np.random.permutation(m)
        x = x[permutation]
        y = y[permutation] 
        
        print('Train data shape: ',x.shape)         

    elif (data == 'android'):

        path = './datasets/android'
        data = pd.read_csv(os.path.join(path,'android.csv'))

        x = data.iloc[:,:-1].copy()
        y = data['Result'].copy()          
        
        x = x.to_numpy()
        y = y.to_numpy()
 
        m = x.shape[0]
        np.random.seed(0)
        permutation = np.random.permutation(m)
        x = x[permutation]
        y = y[permutation] 
        
        print('Train data shape: ',x.shape)

    elif (data == 'auction'):

        path = './datasets/auction'
        data = pd.read_csv(os.path.join(path,'auction.csv'))

        x = data.iloc[:,:-2].copy()
        y = data['verification.result'].copy()          
        
        x = x.to_numpy()
        y = y.to_numpy()
        
        m = x.shape[0]
        np.random.seed(0)
        permutation = np.random.permutation(m)
        x = x[permutation]
        y = y[permutation] 
        
        print('Train data shape: ',x.shape)

    elif (data == 'heart_failure'):

        path = './datasets/heart_failure'
        data = pd.read_csv(os.path.join(path,'heart_failure.csv'))

        x = data.iloc[:,:-1].copy()
        y = data.iloc[:, -1].copy()         
        
        x = x.to_numpy()
        y = y.to_numpy()
        
        m = x.shape[0]
        np.random.seed(0)
        permutation = np.random.permutation(m)
        x = x[permutation]
        y = y[permutation] 
        
        print('Train data shape: ',x.shape)       
        
    return x, y