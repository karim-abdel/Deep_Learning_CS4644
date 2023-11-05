import numpy as np

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        '''
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels,  self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        '''
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        '''
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################
        
        padd = self.padding
        padded = np.pad(x, [(0,0), (0,0), (padd,padd), (padd,padd)], 'constant')
        stride = self.stride
        Q, d, ee, rr = self.weight.shape
        t, d, a, d = x.shape
      
        hh = int(1 + (a + 2 * padd - ee) / stride)
       
        pp = int(1 + (d + 2 * padd - rr) / stride)

        

        out = np.zeros((t, Q, hh, pp))
        for tt in range(t): # 
          for pp in range(Q): 
            for qq in range(0,a+2*padd-ee+1,stride):
              for ff in range(0,d+2*padd-rr+1,stride):
                out[tt, pp, int(qq/stride), int(ff/stride)] = np.sum(np.multiply(padded[tt,:,qq:qq+ee,ff:ff+rr],self.weight[pp,:,:,:])) + self.bias[pp]

        
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        '''
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        '''
        x = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################
        
        padd = self.padding
        dx = np.zeros_like(x)
        
        db = np.zeros_like(self.bias)

        dw = np.zeros_like(self.weight)
        padded = np.pad(x, [(0,0), (0,0), (padd,padd), (padd,padd)], 'constant')
        padded_dx = np.pad(dx, [(0,0), (0,0), (padd,padd), (padd,padd)], 'constant')
        stride = self.stride
        q, e, oo, pp = self.weight.shape
        a, e, o, p = x.shape
        ll = int(1 + (o + 2 * padd - oo) / stride)
        ww = int(1 + (p + 2 * padd - pp) / stride)

        for tt in range(a): 
          for ff in range(q): 
            for rr in range(ll):
              hs = rr * stride
              for kk in range(ww):
                ws = kk * stride
                window = padded[tt, :, hs:hs+oo, ws:ws+pp]
                db[ff] += dout[tt, ff, rr, kk]
                padded_dx[tt, :, hs:hs+oo, ws:ws+pp] += self.weight[ff] * dout[tt, ff, rr, kk]
                dw[ff] += window*dout[tt, ff, rr, kk]

        dx = padded_dx[:, :, padd:padd+o, padd:padd+p]

        self.dx = dx
        
        self.db = db
        self.dw = dw
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################