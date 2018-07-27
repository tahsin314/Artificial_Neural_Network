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

"""
You have to assign train and test data and labels to nn
Since we don't have to train anything, we assigned same data to
train and test
"""
nn = nn(X.T, y.T, X.T, y.T)
nn.model(restore_model='MNIST')
pred = nn.predict(test_data=X.T).T
print(pred.shape)

# Final predictions
print(np.argmax(pred, axis=1))