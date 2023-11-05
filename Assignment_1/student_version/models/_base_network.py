# Do not use packages that are not in standard distribution of python
import numpy as np
class _baseNetwork:
    def __init__(self, input_size=28 * 28, num_classes=10):

        self.input_size = input_size
        self.num_classes = num_classes

        self.weights = dict()
        self.gradients = dict()

    def _weight_init(self):
        pass

    def forward(self):
        pass

    def softmax(self, scores):
        '''
        Compute softmax scores given the raw output from the model

        :param scores: raw scores from the model (N, num_classes)
        :return:
            prob: softmax probabilities (N, num_classes)
        '''
        
        #############################################################################
        # TODO:                                                                     #
        #    1) Calculate softmax scores of input images                            #
        #############################################################################
        ez = np.ez(scores-np.max(scores))
        prob = ez / np.sum(ez,axis=1,keepdims=True)
        return prob
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        
        

    def cross_entropy_loss(self, x_pred, y):
        
        '''
        Compute Cross-Entropy Loss based on prediction of the network and labels
        :param x_pred: Raw prediction from the two-layer net (N, num_classes)
        :param y: Labels of instances in the batch
        :return: The computed Cross-Entropy Loss
        '''
        
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement Cross-Entropy Loss
                                           #
        #m = y.shape[0]
        
        one_h = np.zeros((x_pred.shape[0], x_pred.shape[1]))
        count = 0
        m = x_pred.shape[0]
        for k in y:
          one_h[count, k] = 1
          count += 1
        loss = -np.sum(one_h*np.log(x_pred))/m
        
        #############################################################################
        

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss

    def compute_accuracy(self, x_pred, y):
        y = np.array(y)
        '''
        Compute the accuracy of current batch
        :param x_pred: Raw prediction from the two-layer net (N, num_classes)
        :param y: Labels of instances in the batch
        :return: The accuracy of the batch
        '''
        acc = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the accuracy function                                     #
        #############################################################################
        d = y.shape[0]
        predictions = np.argmax(x_pred, axis=1)
        acc = np.sum(predictions == y) / d
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return acc

    def sigmoid(self, X):
        '''
        Compute the sigmoid activation for the input

        :param X: the input data coming out from one layer of the model (N, num_classes)
        :return:
            out: the value after the sigmoid activation is applied to the input (N, num_classes)
        '''
        #out = None
        out = np.zeros(X.shape)
        temp = np.zeros(X[0].shape)
        
        #############################################################################
        # TODO: Comput the sigmoid activation on the input                          #
        #############################################################################
        for i in range(len(X)):
          u = 0
          for j in X[i]:
            k = (1/(1 + np.exp(-j)))
            temp[u] = float(k) 
            u += 1
          out[i] = temp
          temp = np.zeros(X[0].shape)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        
        assert type(out) != None
        return out

    def sigmoid_dev(self, x):
        
        '''
        The analytical derivative of sigmoid function at x
        :param x: Input data
        :return: The derivative of sigmoid function at x
        '''
        #ds = None
        ds = np.zeros(x.shape)
        temp = np.zeros(x[0].shape)
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the derivative of Sigmoid function                        #
        #############################################################################
        for i in range(len(x)):
          u = 0
          for j in x[i]:
            k = (1/(1 + np.exp(-j)))
            dv = k*(1-k)
            temp[u] =float(dv)
            u+=1
          ds[i]=temp
          temp = np.zeros(x[0].shape)
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        
        return ds

    def ReLU(self, X):
        '''
        Compute the ReLU activation for the input

        :param X: the input data coming out from one layer of the model (N, num_classes)
        :return:
            out: the value after the ReLU activation is applied to the input (N, num_classes)
        '''
        #out = None
        #temp = np.zeros(X[0].shape)
        #############################################################################
        # TODO: Comput the ReLU activation on the input                          #
        #############################################################################
        out = X * (X > 0)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        #assert type(out) != None
        
        return out

    def ReLU_dev(self,X):
        '''
        Compute the gradient ReLU activation for the input

        :param X: the input data coming out from one layer of the model (N, num_classes)
        :return:
            out: gradient of ReLU given input X
        '''
        #out = None
        #out = np.zeros(X.shape)
        #temp = np.zeros(X[0].shape)
        #############################################################################
        # TODO: Comput the gradient of ReLU activation                              #
        #############################################################################
        out = 1. * (X > 0)
       
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        #assert type(out) != None
        
        return out
