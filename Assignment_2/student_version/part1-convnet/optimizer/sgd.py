from ._base_optimizer import _BaseOptimizer
class SGD(_BaseOptimizer):
    def __init__(self, model, learning_rate=1e-4, reg=1e-3, momentum=0.9):
        super().__init__(model, learning_rate, reg)
        self.momentum = momentum

        # initialize the velocity terms for each weight
        self.v_ww = {}


        self.v_bb = {}
        
        for idx,m in enumerate(model.modules):
          if hasattr(m,'weight'):

            self.v_ww[m] = [0.0 for jj in range(m.weight.shape[1]) for kk in range (m.weight.shape[0])]

          if hasattr(m,'bias'):
            self.v_bb[m] = [0.0 for yy in range(m.bias.shape[0])]

    def update(self, model):
        '''
        Update model weights based on gradients
        :param model: The model to be updated
        :return: None, but the model weights should be updated
        '''
        #self.apply_regularization(model)
        
        
        for idx, m in enumerate(model.modules):
            
            if hasattr(m, 'weight'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for weights                                        #
                #############################################################################
                dw= m.dw
                
                
                self.v_ww[m] = [self.momentum * v - self.learning_rate * dp for v,dp in zip(self.v_ww[m],dw)]


                m.weight += self.v_ww[m]
                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################
            if hasattr(m, 'bias'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for bias                                           #
                #############################################################################
                db= m.db

                
                self.v_bb[m] = [self.momentum * v - self.learning_rate * dp for v,dp in zip(self.v_bb[m],db)]
                
                m.bias += self.v_bb[m]

                #v_b[idx] = self.momentum * v_b[max(0,idx-1)] - self.learning_rate * self.db #check correctness
                #self.bias = self.bias + v_w[idx]
                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################
