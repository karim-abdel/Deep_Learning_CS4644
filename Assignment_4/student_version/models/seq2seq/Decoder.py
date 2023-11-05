"""
S2S Decoder model.  (c) 2021 Georgia Tech

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

import random

import torch
import torch.nn as nn
import torch.optim as optim


class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """

    def __init__(self, emb_size, encoder_hidden_size, decoder_hidden_size, output_size, dropout=0.2, model_type="RNN"):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.model_type = model_type

        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the decoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer, this part is controlled by the "model_type"   #
        #          argument. You need to support the following type(in string):     #
        #          "RNN", "LSTM".                                                   #
        #       3) A single linear layer with a (log)softmax layer for output       #
        #       4) A dropout layer                                                  #
        #                                                                           #
        # NOTE: Use nn.RNN and nn.LSTM instead of the naive implementation          #
        #############################################################################
        
        
        self.emb = nn.Embedding(self.output_size, self.emb_size)

        if model_type == "LSTM":
            self.rnn = nn.LSTM(self.emb_size,self.decoder_hidden_size,batch_first=True)



        if model_type == "RNN":
            self.rnn = nn.RNN(self.emb_size,self.decoder_hidden_size,batch_first=True)
        

        self.lin = nn.Linear(self.decoder_hidden_size,self.output_size)
        self.logmax = nn.LogSoftmax(dim=-1)


        self.drop = nn.Dropout(dropout)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def compute_attention(self, hidden, encoder_outputs):
        """ compute attention probabilities given a controller state (hidden) and encoder_outputs using cosine similarity
            as your attention function.

                cosine similarity (q,K) =  q@K.Transpose / |q||K|
                hint |K| has dimensions: N, T
                Where N is batch size, T is sequence length

            Args:
                hidden (tensor): the controller state (dimensions: 1,N, hidden_dim)
                encoder_outputs (tensor): the outputs from the encoder used to implement attention (dimensions: N,T, hidden dim)
            Returns:
                attention: attention probabilities (dimension: N,1,T)
        """

        #############################################################################
        #                              BEGIN YOUR CODE                              #
        # DO NOT USE nn.torch.functional.cosine_similarity or some other library    #
        # function. Implement from formula given in docstring directly              #
        #############################################################################

        q_K = torch.einsum("nh,nth->nt", hidden.squeeze(0), encoder_outputs) 
        q_norm = torch.norm(hidden.squeeze(0), dim=-1)
        K_norm = torch.norm(encoder_outputs, dim=-1)
        cos_sim = q_K / (q_norm.unsqueeze(-1) * K_norm)
    
        
        attention = torch.nn.functional.softmax(cos_sim, dim=-1).unsqueeze(1)
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return attention

    def forward(self, input, hidden, encoder_outputs=None, attention=False):
        """ The forward pass of the decoder
            Args:
                input (tensor): the encoded sequences of shape (N, 1). HINT: encoded does not mean from encoder!!
                hidden (tensor): the hidden weights of the previous time step from the decoder, dimensions: (1,N,decoder_hidden_size)
                encoder_outputs (tensor): the outputs from the encoder used to implement attention, dimensions: (N,T,encoder_hidden_size)
                attention (Boolean): If True, need to implement attention functionality
            Returns:
                output (tensor): the output of the decoder, dimensions: (N, output_size)
                hidden (tensor): the weights coming out of the hidden unit, dimensions: (1,N,decoder_hidden_size)
            where N is the batch size, T is the sequence length
        """

        #############################################################################
        # TODO: Implement the forward pass of the decoder.                          #
        #       Apply the dropout to the embedding layer before you apply the       #
        #       recurrent layer                                                     #
        #                                                                           #
        #       if attention is true, compute the attention probabilities and use   #
        #       it to do a weighted average on the hidden and cell states to        #
        #       determine what will be consumed by the recurrent layer              #
        #                                                                           #
        #       Apply linear layer and softmax activation to output tensor before   #
        #       returning it.                                                       #
        #############################################################################
        emb = self.drop(self.emb(input))

        if self.model_type == "LSTM":
          hidden, c = hidden

        if attention:
          att = self.compute_attention(hidden, encoder_outputs)
          hidden = att @ encoder_outputs
          hidden = hidden.transpose(0,1)


          if self.model_type == "LSTM":
            att = self.compute_attention(c, encoder_outputs)
            cell = att @ encoder_outputs
            c=cell.transpose(0,1)
          
        if self.model_type == "LSTM":
          hidden = hidden, c


        out, hidden = self.rnn(emb, hidden)

        out = out.transpose(0,1).squeeze(0)
        
        output = self.logmax(self.lin(out))
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return output, hidden
