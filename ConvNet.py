import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()
        
        # This will select the forward pass function based on mode for the ConvNet.
        # Based on the question, you have 5 modes available for step 1 to 5.
        # During creation of each ConvNet model, you will assign one of the valid mode.
        # This will fix the forward function (and the network graph) for the entire training/testing
        if mode == 1:
            # Convolutional layers           
            self.conv1 = nn.Conv2d(3, 40, 5)   # First conv layer
            self.conv2 = nn.Conv2d(40, 40, 5)  # Second conv layer

            # Fully connected layers
            self.fc1 = nn.Linear(40 * 4 * 4, 100)  # First FC layer
            self.fc2 = nn.Linear(100, 100)  # Second (new) FC layer w/ 100 neurons
            self.fc3 = nn.Linear(100, 24)  # Output layer for 10 classes 
            self.forward = self.model_1

        elif mode == 2:
            # Convolutional layers           
            self.conv1 = nn.Conv2d(3, 40, 5)   # First conv layer
            self.conv2 = nn.Conv2d(40, 40, 5)  # Second conv layer

            self.fc1 = nn.Linear(40 * 4 * 4, 1000)  # First FC layer
            self.fc2 = nn.Linear(1000, 1000)  # Changed second FC layer w/ 1000 neurons
            self.fc3 = nn.Linear(1000, 24)  # Output layer for 10 classes

            self.dropout = nn.Dropout(p=0.5) # Dropout for regularization
            self.forward = self.model_2

        else: 
            print("Invalid mode ", mode, "selected. Select between 1-2")
            exit(0)
        

    # Add one extra fully connected layer.
    def model_1(self, X):
        # First convolutional layer with ReLU activation
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, kernel_size=2, stride=2)  # 2x2 max pooling
        
        # Second convolutional layer with ReLU activation
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, kernel_size=2, stride=2)  # 2x2 max pooling
        
        # Flatten the output to connect it to the fully connected layers
        X = X.view(X.size(0), -1)
        
        # First fully connected layer with ReLU activation
        X = F.relu(self.fc1(X))
        
        # Second fully connected layer with ReLU activation
        X = F.relu(self.fc2(X))
        
        # Output layer
        fcl = self.fc3(X)
        return fcl

    # Add Dropout now.
    def model_2(self, X):
        # First convolutional layer with ReLU activation
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, kernel_size=2, stride=2)  # 2x2 max pooling
        
        # Second convolutional layer w/ ReLU activation 
        X = F.relu(self.conv2(X))                             
        X = F.max_pool2d(X, kernel_size=2, stride=2)  # 2x2 max pooling
        
        # Flatten the output 
        X = X.view(X.size(0), -1)               
        
        # First fully connected layer w/ ReLU activation
        X = F.relu(self.fc1(X))           
        
        # Apply first Dropout
        X = self.dropout(X)                  
        
        # Second fully connected layer w/ ReLU activation
        X = F.relu(self.fc2(X))   
        
        # Apply second Dropout
        X = self.dropout(X)    
        
        # Output layer 
        fcl = self.fc3(X)      
        return fcl