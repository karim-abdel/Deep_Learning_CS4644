import torch
import torch.nn as nn


class MyModel(nn.Module):
    # You can use pre-existing models but change layers to recieve full credit.
    def __init__(self):
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################
        #use sequential so forwards pass is easier !!!
        
        self.mymodell = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            
            nn.ReLU(),
            
            nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1),
            
            nn.MaxPool2d(2, 2), 
            
            nn.BatchNorm2d(128),

            #check output is correct
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            
            nn.ReLU(),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            
            nn.ReLU(),
            
            nn.MaxPool2d(2, 2), 
            
            nn.BatchNorm2d(128),
            #check output is correct
            
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            
            nn.ReLU(),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            
            nn.ReLU(),
            
            nn.MaxPool2d(2, 2), 
            
            nn.BatchNorm2d(128),

            #check output is correct
            
            nn.Flatten(), 
            
            nn.Linear(2048, 1024),
            
            nn.ReLU(),
            
            nn.Linear(1024, 512),
            
            
            nn.ReLU(),
            
            nn.Linear(512, 10))
        
          #check output is correct
   

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        
        outs=self.mymodell(x)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs