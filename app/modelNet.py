import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

netTransform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])

class Net(nn.Module):
    """
    Convolutional Neural Network for MNIST digit classification (0-9).
    
    Architecture:
        - Two convolutional layers with ReLU activation and max pooling
        - Two fully connected layers with dropout regularization
        - Log-softmax output for multiclass classification
    
    Input: Grayscale images of shape (batch_size, 1, 28, 28)
    Output: Log probabilities of shape (batch_size, 10)
    
    The network follows a standard CNN pattern: feature extraction via 
    convolutions followed by classification via fully connected layers.
    """
    
    def __init__(self):
        """
        Initialize the CNN architecture with convolutional and linear layers.
        
        Layer specifications:
            conv1: 1 -> 10 channels, 5x5 kernel - detects basic features (edges, lines)
            conv2: 10 -> 20 channels, 5x5 kernel - detects complex patterns (shapes, curves)
            conv2_drop: 2D dropout with p=0.5 - regularization for convolutional features
            fc1: 320 -> 50 neurons - feature integration layer
            fc2: 50 -> 10 neurons - classification layer (one per digit class)
        
        Note: The input dimension 320 for fc1 is calculated as:
              20 channels * 4 * 4 pixels = 320 (after convolutions and pooling)
        """
        super(Net, self).__init__()
        
        # Convolutional layers for hierarchical feature extraction
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(in_features=320, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=10)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            torch.Tensor: Log probabilities of shape (batch_size, 10)
            
        Data flow:
            Input (1,28,28) -> Conv1+ReLU+MaxPool -> (10,12,12)
            -> Conv2+Dropout+ReLU+MaxPool -> (20,4,4) -> Flatten -> (320,)
            -> FC1+ReLU+Dropout -> (50,) -> FC2+LogSoftmax -> (10,)
        """
        # First convolutional block: feature detection
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        # Shape: (batch_size, 10, 12, 12)
        
        # Second convolutional block: pattern recognition with regularization
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), kernel_size=2))
        # Shape: (batch_size, 20, 4, 4)
        
        # Flatten spatial dimensions for fully connected layers
        x = x.view(-1, 320)
        # Shape: (batch_size, 320)
        
        # First fully connected layer with dropout regularization
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        # Shape: (batch_size, 50)
        
        # Final classification layer
        x = self.fc2(x)
        # Shape: (batch_size, 10)
        
        return F.log_softmax(x, dim=1)