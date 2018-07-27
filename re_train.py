from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from Artificial_Neural_Network import NeuralNetwork as nn
import numpy as np
import os

"""
If you don't have mnist dataset downloaded, it will download
and store it inside a newly created directory 'data'
"""
if not os.path.isdir('data'):
    os.mkdir('data')
mnist = fetch_mldata('MNIST original', data_home='data')
X, y = mnist.data/255, mnist.target
X, y = shuffle(X, y)

"""
Converting labels to hot vetors
"""
hot_vec = np.zeros((len(y), int(np.max(y))+1))
for i in range(len(hot_vec)):
    hot_vec[i, int(y[i])] = 1
y = hot_vec

# 85% data for train and 15% data for test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_test, y_train, y_test = X_train.T, X_test.T, y_train.T, y_test.T
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Code for training from scratch and saving model starts here
nn = nn(X_train, y_train, X_test, y_test)
nn.model(restore_model='MNIST')  # Model restored
nn.build_model(batch_size=1000, final_activation='sigmoid', epoch=100, lr=0.03)
nn.compile(save_model='MNIST')  # Saving model
