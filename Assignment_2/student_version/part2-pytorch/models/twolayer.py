import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        '''
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        '''
        super(TwoLayerNet, self).__init__()
        #############################################################################
        # TODO: Initialize the TwoLayerNet, use sigmoid activation between layers   #
        #############################################################################
        
        self.fcc1 = nn.Linear(input_dim, hidden_size)

        self.sigmoid=nn.Sigmoid()
        #double check
        self.fcc2 = nn.Linear(hidden_size, num_classes)

        self.softmax=nn.Softmax(dim=1)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        out1=self.fcc1(x.view(x.shape[0],-1))

        out2=self.sigmoid(out1)

        out3=self.fcc2(out2)
        #double check
        out=self.softmax(out3)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out