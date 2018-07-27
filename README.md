# Artificial Neural Network

This repository contains `Artificial_Neural_Network.py` -a generic code for *Neural Network*. Traditionally, if you want to 
write a code for a neural network from scratch, you need to know number of layers and activations in 
each layer. Also you need to write code for both forward and backward propagation.But using my script, you'll only need to write the 
forward propagation part and the script will take care of the rest. You have to define a network, number of neurons in each layer and corresponding 
activation functions in those layers. 
### Requirements:
    1. Numpy
    2. Sklearn (If you want to use datasets from sklearn)
    3. Tqdm (For progress bar)

### Features:
    1. Optimizer: Gradient descent (More optimizers will be
     added later)
    2. Loss function: Cross-entropy
    3. Regularizations: None (Maybe later sometime I'll add Dropout 
    and L2-reg)
    4. Activations: Sigmoid, tanh, ReLU and softmax
    5. Saving and restoring Model (including model summary 
    and weights)  

## How to use:
1. __Feeding data:__ You have to feed train and test data and labels as hot vector to the class.
Example:
    ~~~~
    import Artificial_Neural_Network as NN
    nn = NN(train_data, train_label, test_data, test_label)
    ~~~~
    train and test data format should be `(number of data, features)`. For example:  If you have
    60000 data for train set and 10000 data for test set, each data has 784 features (MNIST flattend) 
    and 10 labels, then 
    ~~~~
    train_data shape: (784, 60000)
    test_data shape: (784, 10000)
    train_label shape: (10, 60000)
    test_label shape: (10, 10000)
    ~~~~
2. __Adding layer:__ Layer can be added using `dense` function. 
    ~~~~
    nn.dense(neurons, activation='sigmoid', layer_name=None)
    ~~~~
    You have to provide number of *neurons*, *activation* function and *layer name* (optional).
    Example: 
    ~~~~
    nn.dense(100, activation='relu', layer_name='ReLU layer')
    ~~~~ 
    
    It adds a layer with 100 neurons and use *`ReLU`* as activation,.
    
3. __Finalize your model:__ After you are finished with adding dense layers, you have to define *final activation*, *cost_function* and
    *optimizer*. Also you need to define batch size `batch_size`, number of epochs `epoch` and learning rate `lr`. You can do them using `build_model`
    ~~~~
    nn.build_model(self, batch_size, final_activation='sigmoid', cost_function='binary_cross-entropy', epoch=100, optimizer='gradient_descent', lr=0.1)
    ~~~~ 
    
    Example: 
    
    ~~~~
    nn.build_model(batch_size=10000, final_activation='softmax', epoch=100, lr=0.25)
    ~~~~
4. __Training:__ You can train your model using `compile`.

    ~~~~
    nn.compile(save_model=None)
    ~~~~
    
    You can save your model with a name by setting `save_model=model_name`
    
    Example:
    
    ~~~~
    nn.compile(save_model='my_model')
    ~~~~
    
5. __Prediction__: You can predict over test data using `predict` function. It returns predictions as one hot vector.
    ~~~~
    nn.predict(self, test_data=None)
    ~~~~ 
    
    Example:
    
    ~~~~
    nn.predict(self, test_data=test_data)
    ~~~~
    
    `test_data` has to be of shape `(number of data, features)`
6. __Restoring saved model:__ You can restore saved model for prediction or re-training using `model`

    ~~~~
    nn.model(restore_model=None)
    ~~~~ 
    
    Example:
    
    ~~~~
    nn.model(restore_model='my_model')
    ~~~~
    If you restore model, you don't need to define `dense` layers. It will automatically
    load architecture from the model.
## Examples: 
1.  __Training from scratch:__ See `train.py`
2. __Predicting using saved model:__ See `predict.py`
3. __Re-training using saved model:__ See `re_train.py`