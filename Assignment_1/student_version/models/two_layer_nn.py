# Do not use packages that are not in standard distribution of python
import numpy as np
np.random.seed(1024)
from ._base_network import _baseNetwork

class TwoLayerNet(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10, hidden_size=128):
        super().__init__(input_size, num_classes)

        self.hidden_size = hidden_size
        self._weight_init()


    def _weight_init(self):
        '''
        initialize weights of the network
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the first layer of shape (num_features, hidden_size)
        - b1: The bias term of the first layer of shape (hidden_size,)
        - W2: The weight matrix of the second layer of shape (hidden_size, num_classes)
        - b2: The bias term of the second layer of shape (num_classes,)
        '''

        # initialize weights
        self.weights['b1'] = np.zeros(self.hidden_size)
        self.weights['b2'] = np.zeros(self.num_classes)
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.hidden_size)
        np.random.seed(1024)
        self.weights['W2'] = 0.001 * np.random.randn(self.hidden_size, self.num_classes)

        # initialize gradients to zeros
        self.gradients['W1'] = np.zeros((self.input_size, self.hidden_size))
        self.gradients['b1'] = np.zeros(self.hidden_size)
        self.gradients['W2'] = np.zeros((self.hidden_size, self.num_classes))
        self.gradients['b2'] = np.zeros(self.num_classes)

    def forward(self, X, y, mode='train'):
        X = np.array(X)
        y = np.array(y)
        '''
        The forward pass of the two-layer net. The activation function used in between the two layers is sigmoid, which
        is to be implemented in self.,sigmoid.
        The method forward should compute the loss of input batch X and gradients of each weights.
        Further, it should also compute the accuracy of given batch. The loss and
        accuracy are returned by the method and gradients are stored in self.gradients

        :param X: a batch of images (N, input_size)
        :param y: labels of images in the batch (N,)
        :param mode: if mode is training, compute and update gradients;else, just return the loss and accuracy
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
            self.gradients: gradients are not explicitly returned but rather updated in the class member self.gradients
        '''
        loss = None
        accuracy = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process:                                      #
        #        1) Call sigmoid function between the two layers for non-linearity  #
        #        2) The output of the second layer should be passed to softmax      #
        #        function before computing the cross entropy loss                   #
        #    2) Compute Cross-Entropy Loss and batch accuracy based on network      #
        #       outputs                                                             #
        #############################################################################
        z1 = np.dot(X, self.weights['W1']) + self.weights['b1']
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, self.weights['W2']) + self.weights['b2']
        a2 = self.softmax(z2)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        loss =  self.cross_entropy_loss(a2, y)
        
        accuracy = self.compute_accuracy(a2, y)
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the backward process:                                     #
        #        1) Compute gradients of each weight and bias by chain rule         #
        #        2) Store the gradients in self.gradients                           #
        #    HINT: You will need to compute gradients backwards, i.e, compute       #
        #          gradients of W2 and b2 first, then compute it for W1 and b1      #
        #          You may also want to implement the analytical derivative of      #
        #          the sigmoid function in self.sigmoid_dev first                   #
        #############################################################################
        
        
        
        one_h = np.zeros((a2.shape[0], a2.shape[1]))
        count = 0
        for k in y:
          one_h[count, k] = 1
          count += 1
        
        

        if mode == 'train':
      
            dz2 = a2 - one_h
            dw2 = np.dot(a1.T, dz2) / X.shape[0]
            db2 = np.sum(dz2, axis=0) / X.shape[0]

            da1 = np.dot(dz2, self.weights['W2'].T)
            dz1 = da1 * self.sigmoid_dev(z1)
            dw1 = np.dot(X.T, dz1) / X.shape[0]
            db1 = np.sum(dz1, axis=0) / X.shape[0]

           
            self.gradients['W2'] = dw2
            self.gradients['b2'] = db2
            self.gradients['W1'] = dw1
            self.gradients['b1'] = db1

     
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


        return loss, accuracy


