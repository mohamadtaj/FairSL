import tensorflow as tf
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from tensorflow.keras import datasets, models, Model, Input
from tensorflow.keras import regularizers

def nn_model (dataset, input_size, output_size):
    if (dataset == 'cifar10'):
        return model_cifar(input_size, output_size)
    elif (dataset == 'reuters'):
        return model_reuters(output_size)        
      
  
def model_cifar(input_size, output_size):

    initializer = tf.keras.initializers.he_uniform()    
    model = models.Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=initializer, padding='same', input_shape = input_size))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=initializer, padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializer, padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializer, padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer=initializer, padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer=initializer, padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer=initializer))
    model.add(Dropout(0.5))     
    model.add(Dense(output_size, activation='softmax'))
    
    return model


def model_reuters(output_size):   

    model = models.Sequential()
    model.add(Embedding(input_dim=10000, output_dim=256, input_length=500))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(output_size, activation='softmax'))  
    
    return model    