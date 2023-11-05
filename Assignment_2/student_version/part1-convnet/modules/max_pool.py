import numpy as np

class MaxPooling:
    '''
    Max Pooling of input
    '''
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        '''
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        '''
        out = None
        
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        uu = 2
        yy = 2
        stride = self.stride
        f, a, e, y = x.shape
        pp = int(1 + (e - uu) / stride)
        
        zz = int(1 + (y - yy) / stride)

        out = np.zeros((f, a, pp, zz))

        for qq in range(f):

          for rr in range(a):

            for ff in range(pp):

              hs = ff * stride

              for ss in range(zz):

                ws = ss * stride

        
                window = x[qq, rr, hs:hs+uu, ws:ws+yy]


                out[qq, rr, ff, ss] = np.max(window)

        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, hs,ws)
        return out

    def backward(self, dout):
        '''
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return:
        '''
        x, H_out, W_out = self.cache
        
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                     #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################
        
        zz = 2
        aa = 2
        stride = self.stride
        f, s, k, b = x.shape
        pp = int(1 + (k - zz) / stride)
        rr = int(1 + (b - aa) / stride)

        self.dx = np.zeros_like(x)



        for tt in range(f):
          
          for yy in range(s):

            for mm in range(pp):

              hs = mm * stride

              for er in range(rr):
                ws = er * stride

                # remember to check if is correct
                window = x[tt, yy, hs:hs+zz, ws:ws+aa]
                m = np.max(window)


                
                self.dx[tt, yy, hs:hs+zz, ws:ws+aa] += (window == m) * dout[tt, yy, mm, er]

        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
