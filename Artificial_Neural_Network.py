import numpy as np
from tqdm import tqdm
import pandas as pd
np.random.seed(42)


class NeuralNetwork:
    """
     This is going to be an awesome class. This will be a generalized
    class for training neural network. You'll only need to write the
    forward propagation part using this. The class and it's children and
     relative function will take care of the back propagation part.

    """

    def __init__(self, x_train, y_train, x_test, y_test):
        # Initiating train and test data
        self.X_train = x_train
        self.y_train = y_train
        self.X_test = x_test
        self.y_test = y_test.T
        # Defining parameters
        self.parameters = {}
        self.parameters.update({'layer_number': 0})
        self.parameters.update({'neurons': []})
        self.parameters.update({'activation': []})
        self.parameters.update({'layer name': []})
        # Layer outputs
        self.A = {}
        # Weights and biases
        self.W = {}
        self.b = {}
        self.d = {}
        # Derivatives of weights and biases
        self.dW = {}
        self.db = {}
        # Epoch, optimizer and learning rate with default values
        self.epoch = 100
        self.optimizer = None
        self.learning_rate = None
        self.cost_function = None
        self.Cost = None
        # Batch size
        self.batch_size = 1

        # Model and summary
        self.model_restored = False
        self.parameters.update({'Summary': None})

    # Activation functions
    def sigmoid(self, z):
        """
        Returns sigmoid output
        :param z: input from previous layer
        :return: sigmoid(z)
        """
        return 1/(1+np.exp(-z))

    def tanh(self, z):
        """
        Returns tan hyperbolic output
        :param z: input from previous layer
        :return: tanh(z)
        """
        return np.tanh(z)

    def relu(self, z):
        """
        Returns rectified linear unit(ReLU) output
        :param z: input from previous layer
        :return: ReLU(z)
        """
        return np.maximum(0, z)

    def softmax(self, z):
        """
        Returns softmax output
        :param z: input from previous layer
        :return: softmax(z)
        """
        exp_scores = np.exp(z)
        return exp_scores / np.sum(exp_scores, axis=0, keepdims=True)

    # Derivative of acivation functions
    def dsigmoid(self, z):
        """ Returns derivative of sigmoid function """
        return np.multiply(z, 1-z)

    def dtanh(self, z):
        """ Returns derivative of tanh function """
        return 1 - np.multiply(z, z)

    def drelu(self, z):
        """ Returns derivative of ReLU function """
        z[z <= 0] = 0
        z[z > 0] = 1
        return z

    def dsoftmax(self, z, y):
        """ Returns derivative of softmax output. Softmax works
         only in the final layer. Derivative of softmax requires
          final output and labels provided as one-hot vector. """
        return (z - y)/y.shape[0]

    def cost(self, z, y, name='binary_cross_entropy'):
        """ Returns cost """
        if name == 'binary_cross_entropy':
            loss = -(np.multiply(y, np.log(z)) + np.multiply((1 - y), np.log(1 - z)))
            c = np.sum(loss)/y.shape[1]
            return c

    def model(self, restore_model=None):
        """ This is for saving and restoring model """
        if restore_model is not None:
            self.model_restored = True
            Model = np.load(restore_model+'.npz')
            self.parameters = Model['mdict'].item()
            print('Model Restored Successfully')

    def dense(self, neurons, activation='sigmoid', layer_name=None):
        """ Calling this function would just update information
                about the architecture of the network """

        self.parameters['layer_number'] += 1
        if layer_name is not None:
            self.parameters['layer name'].append(layer_name)
        else:
            self.parameters['layer name'].append('layer'+str(self.parameters['layer_number']))
        layer = self.parameters['layer_number']
        self.parameters['neurons'].append(neurons)
        self.parameters['activation'].append(activation)

        if layer == 1:
            w = np.random.randn(neurons, self.X_train.shape[0])*0.1
        else:

            prev_neurons = self.parameters['neurons'][layer - 2]
            w = np.random.randn(neurons, prev_neurons) * 0.1
        b = np.random.randn(neurons, 1) * 0.1

        self.parameters.update({'W' + str(layer): w})
        self.parameters.update({'b' + str(layer): b})

    def forward_propagation(self, a, w, b, activation='sigmoid'):
        """ Function name is self explanatory. """
        a_new = np.dot(w, a) + b
        if activation == 'sigmoid':
            a_new = self.sigmoid(a_new)
        elif activation == 'tanh':
            a_new = self.tanh(a_new)
        elif activation == 'relu':
            a_new = self.relu(a_new)
        elif activation == 'softmax':
            a_new = self.softmax(a_new)
        return a_new

    def build_model(self, batch_size, final_activation='sigmoid', cost_function='binary_cross-entropy', epoch=100, optimizer='gradient_descent', lr=0.1):
        """ After defining the network, you have to call this function. This
         function finalizes your network with necessary parameters. Also this
                                function initiates training. """
        self.epoch = epoch
        self.optimizer = optimizer
        self.learning_rate = lr
        self.cost_function = cost_function
        self.batch_size = batch_size
        self.optimizer = optimizer
        layer = self.parameters['layer_number']
        if not self.model_restored:
            self.parameters['activation'].append(final_activation)

            neurons = self.parameters['neurons'][-1]
            w = np.random.randn(self.y_test.shape[1], neurons) * 0.1
            b = np.random.randn(self.y_test.shape[1], 1) * 0.1
            self.parameters.update({'W' + str(layer + 1): w})
            self.parameters.update({'b' + str(layer + 1): b})

        if self.parameters['activation'][-1] == 'tanh':
            print("Warning! You shouldn't place tanh as your "
                  "final layer activation. It won't help your"
                  "model to converge.")

        # Adding summary
        layer_name = self.parameters['layer name']
        neurons = self.parameters['neurons']
        activations = self.parameters['activation']
        trainable_parameters = []
        for i in range(1, layer+2):
            trainable_parameters.append(np.prod(self.parameters['W'+str(i)].shape)+np.prod(self.parameters['b'+str(i)].shape))
        summary = pd.DataFrame({'layer name': layer_name, 'neurons': neurons, 'activations': activations[:-1], 'trainable parameters': trainable_parameters[:-1]},
                               columns=['layer name', 'neurons', 'activations', 'trainable parameters'])
        self.parameters['Summary'] = summary
        summary = str(summary)
        summary += '\nFinal activation: ' + activations[-1]
        summary += ' Total trainables: ' + str(sum(trainable_parameters))
        print(summary)

    def predict(self, test_data=None):
        """ It returns prediction(s)"""
        layer_num = self.parameters['layer_number']
        if test_data is None:
            self.A.update({'A_test0': np.array(self.X_test)})
        else:
            self.A.update({'A_test0': np.array(test_data)})
        for i in range(1, layer_num+2):
            layer_activation = self.parameters['activation'][i-1]
            _A = self.forward_propagation(self.A['A_test'+str(i-1)], self.parameters['W'+str(i)], self.parameters['b'+str(i)], activation=layer_activation)
            self.A.update({'A_test'+str(i): _A})
        return self.A['A_test'+str(layer_num+1)]

    def accuracy(self):
        """
        :return: accuracy
        """
        layer_num = self.parameters['layer_number']
        self.predict()
        acc = np.average(np.argmax(self.A['A_test' + str(layer_num + 1)], axis=0) == np.argmax(self.y_test.T, axis=0))
        return acc

    def update_weight(self):
        """
        It updates weights so that the model converges.
        :return: updated weights
        """
        if self.optimizer == 'gradient_descent':
            for i in range(1, self.parameters['layer_number'] + 2):
                self.parameters['W' + str(i)] -= self.learning_rate * self.dW['dW' + str(i)]
                self.parameters['b' + str(i)] -= self.learning_rate * self.db['db' + str(i)]

    def backward_propagation(self, a, d_next, w_next, activation='sigmoid'):
        """ Self explanatory function name. """
        if activation == 'sigmoid':
            dz = self.dsigmoid(a)
        elif activation == 'tanh':
            dz = self.dtanh(a)
        elif activation == 'relu':
            dz = self.drelu(a)
        d = np.multiply(np.dot(w_next.T, d_next), dz)
        return d

    def batch_formatter(self, start, stop=None):
        """
        Creates Batch
        :param start: start index of data
        :param stop: stop index of data
        :return: A batch of data with length stop-start
        """
        x_ = np.copy(self.X_train.T)
        if stop is not None:
            x = x_[start:stop]
        else:
            x = x_[start:]
        x = x.T
        y_ = np.copy(self.y_train.T)
        if stop is not None:
            y = y_[start:stop]
        else:
            y = y_[start:]
        y = y.T
        n = y.shape[1]
        return x, y, n

    def calculation(self, x, y, n, layer_num):
        """
        This function updates weight using batch data
        :param x: Batch features. dim: [num_feature, batch size]
        :param y: Batch labels. dim: [num_lables, batch_size]
        :param n: batch size
        :param layer_num: Total number of layers
        :return: Calculates cost and updates weights
        """
        # Forward propagation
        self.A.update({'A0': x})
        for i in range(1, layer_num + 2):
            a = self.A['A' + str(i - 1)]
            w = self.parameters['W' + str(i)]
            b = self.parameters['b' + str(i)]
            activation_function = self.parameters['activation'][i - 1]
            self.A.update({'A' + str(i): self.forward_propagation(a, w, b, activation_function)})

        if self.cost_function == 'binary_cross-entropy':
            if self.parameters['activation'][-1] == 'sigmoid':
                self.d['d' + str(layer_num + 1)] = - np.multiply((
                        np.divide(y, self.A['A' + str(layer_num + 1)]) -
                        np.divide(1 - y, 1 - self.A['A' + str(layer_num + 1)])),
                    self.dsigmoid(self.A['A' + str(layer_num + 1)])) / n

            elif self.parameters['activation'][-1] == 'relu':
                self.d['d' + str(layer_num + 1)] = - np.multiply((
                        np.divide(y, self.A['A' + str(layer_num + 1)]) -
                        np.divide(1 - y, 1 - self.A['A' + str(layer_num + 1)])),
                    self.drelu(self.A['A' + str(layer_num + 1)])) / n

            elif self.parameters['activation'][-1] == 'softmax':
                self.d['d' + str(layer_num + 1)] = (self.A['A' + str(layer_num + 1)] - y) / n
            for k in reversed(range(1, layer_num + 1)):
                self.d.update({'d' + str(k): self.backward_propagation(
                    self.A['A' + str(k)], self.d['d' + str(k + 1)],
                    self.parameters['W' + str(k + 1)], self.parameters['activation'][k - 1])})

            for m in reversed(range(1, layer_num + 2)):
                self.dW.update({'dW' + str(m): np.dot(self.d['d' + str(m)], self.A['A' + str(m - 1)].T)})
                d_tmp = self.d['d' + str(m)]
                self.db.update({'db' + str(m): np.sum(d_tmp, axis=1, keepdims=True)})
        # Clipping to avoid zero division
        self.A['A' + str(layer_num + 1)] = np.clip(self.A['A' + str(layer_num + 1)], 1e-6, 1-1e-6)
        self.Cost = self.cost(self.A['A' + str(layer_num + 1)], y)
        self.update_weight()

    def compile(self, save_model=None):
        """
        For training and computing cost and accuracy and prediction.
        :param save_model: Model name that will be saved (optional)
        :return: Creates batches, does forward and backpropagation
        and saves model.
        """
        layer_num = int(self.parameters['layer_number'])
        num_batch = self.X_train.shape[1]//self.batch_size
        for e in tqdm(range(self.epoch)):
            for b in tqdm(range(num_batch)):
                x, y, n = self.batch_formatter(b * self.batch_size, (b + 1)*self.batch_size)
                self.calculation(x, y, n, layer_num)
            x, y, n = self.batch_formatter(num_batch * self.batch_size)
            self.calculation(x, y, n, layer_num)
            if save_model is not None:
                if save_model is not None:
                    np.savez(save_model + '.npz', mdict=self.parameters)
            acc = self.accuracy()
            print('Epoch: ', e, 'Cost: ', self.Cost, 'Accuracy: ', acc)
