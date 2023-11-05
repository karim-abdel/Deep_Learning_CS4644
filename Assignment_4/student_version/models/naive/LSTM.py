"""
LSTM model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import numpy as np
import torch
import torch.nn as nn


class LSTM(nn.Module):
    # An implementation of naive LSTM using Pytorch Linear layers and activations
    # You will need to complete the class init function, forward function and weight initialization

    def __init__(self, input_size, hidden_size):
        """ Init function for LSTM class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
            Returns: 
                None
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        ################################################################################
        # TODO:                                                                        #
        #   Declare LSTM weights and attributes in order specified below to pass GS.   #
        #   You should include weights and biases regarding using nn.Parameter:        #
        #       1) i_t: input gate                                                     #
        #       2) f_t: forget gate                                                    #
        #       3) g_t: cell gate, or the tilded cell state                            #
        #       4) o_t: output gate                                                    #
        #   for each equation above, initialize the weights,biases for input prior     #
        #   to weights, biases for hidden.                                             #
        #   when initializing the weights consider that in forward method you          #
        #   should NOT transpose the weights.                                          #
        #   You also need to include correct activation functions                      #
        ################################################################################

        # i_t: input gate
        self.W_ii = nn.Parameter(torch.Tensor(input_size,hidden_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size,hidden_size))
        self.b_ii = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hi = nn.Parameter(torch.Tensor(hidden_size))

        
        # f_t: the forget gate
        self.W_if = nn.Parameter(torch.Tensor(input_size,hidden_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size,hidden_size))
        self.b_if = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hf = nn.Parameter(torch.Tensor(hidden_size))

        # g_t: the cell gate
        self.W_ig = nn.Parameter(torch.Tensor(input_size,hidden_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size,hidden_size))
        self.b_ig = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hg = nn.Parameter(torch.Tensor(hidden_size))

        # o_t: the output gate
        self.W_io = nn.Parameter(torch.Tensor(input_size,hidden_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size,hidden_size))
        self.b_io = nn.Parameter(torch.Tensor(hidden_size))
        self.b_ho = nn.Parameter(torch.Tensor(hidden_size))

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        self.init_hidden()

    def init_hidden(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: torch.Tensor, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""

        ################################################################################
        # TODO:                                                                        #
        #   Implement the forward pass of LSTM. Please refer to the equations in the   #
        #   corresponding section of jupyter notebook. Iterate through all the time    #
        #   steps and return only the hidden and cell state, h_t and c_t.              # 
        #   Note that this time you are also iterating over all of the time steps.     #
        ################################################################################
     
        bs, seq, f = x.size()
        if init_states is None:
          h_t,c_t = (
                      torch.zeros(bs, self.hidden_size).to(x.device),
                      torch.zeros(bs, self.hidden_size).to(x.device),
                    )
        else:
          h_t,c_t = init_states

        

        for t in range(seq):
          x_t = x[ :, t, : ]
          
          i_t = torch.sigmoid( x_t @ self.W_ii    + self.b_ii +   h_t @ self.W_hi + self.b_hi)
          f_t = torch.sigmoid(x_t @ self.W_if  + self.b_if + h_t @ self.W_hf + self.b_hf)
          g_t = torch.tanh(x_t @ self.W_ig  + self.b_ig + h_t @ self.W_hg + self.b_hg)
          o_t = torch.sigmoid(x_t @ self.W_io + self.b_io + h_t @ self.W_ho + self.b_ho)
          c_t = f_t * c_t + i_t*g_t 
          h_t = o_t * torch.tanh(c_t)



        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return (h_t, c_t)
