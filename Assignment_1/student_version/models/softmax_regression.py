# Do not use packages that are not in standard distribution of python
import numpy as np

from ._base_network import _baseNetwork

class SoftmaxRegression(_baseNetwork):
    def __init__(self, input_size=28*28, num_classes=10):
        '''
        A single layer softmax regression. The network is composed by:
        a linear layer without bias => ReLU activation => Softmax
        :param input_size: the input dimension
        :param num_classes: the number of classes in total
        '''
        super().__init__(input_size, num_classes)
        self._weight_init()

    def _weight_init(self):
        '''
        initialize weights of the single layer regression network. No bias term included.
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the linear layer of shape (num_features, hidden_size)
        '''
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.num_classes)
        self.gradients['W1'] = np.zeros((self.input_size, self.num_classes))

    def forward(self, X, y, mode='train'):
        X = np.array(X)
        y = np.array(y)
       
        '''
        Compute loss and gradients using softmax with vectorization.

        :param X: a batch of image (N, 28x28)
        :param y: labels of images in the batch (N,)
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
        '''
        loss = None
        gradient = None
        accuracy = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process and compute the Cross-Entropy loss    #
        # Hint:                                                                     #
        #   Store your intermediate outputs before ReLU for backwards               #
        Z = np.dot(X,self.weights["W1"])       
        A = self.ReLU(Z)
        p = self.softmax(A)
        
        loss = self.cross_entropy_loss(p,y)
        
        accuracy = self.compute_accuracy(p,y)
        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        if mode != 'train':
            return loss, accuracy

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the backward process:                                     #
        #        1) Compute gradients of each weight and bias by chain rule         #
        #        2) Store the gradients in self.gradients                           #
        #############################################################################
        one_h = np.zeros((p.shape[0], p.shape[1]))
        count = 0
        for k in y:
          one_h[count, k] = 1
          count += 1
        

        s = p - one_h
        r = s*self.ReLU_dev(A)
        
        l = (np.dot(X.T,r))/X.shape[0]
        self.gradients["W1"] = l
        #dw = _baseNetwork.sigmoid_dev(self,E)
        
        #self.weights["W1"] = self.weights["W1"] - w_up
        #self.gradients['W1'] = {'weights': dw}
    
        
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        
        
        return loss, accuracy





        


