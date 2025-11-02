import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

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
    
    def predict(self, images):
        """
        LIME-compatible prediction function for model explanations.
        
        This method is specifically designed to work with LIME (Local Interpretable 
        Model-agnostic Explanations) which requires a prediction function that:
        1. Accepts a list/array of images
        2. Returns probability distributions (not log probabilities)
        3. Handles image preprocessing internally
        
        Args:
            images (list or np.ndarray): List of images as numpy arrays, 
                                       typically of shape (height, width) or (height, width, channels)
                                       
        Returns:
            np.ndarray: Probability distributions of shape (num_images, 10)
                       Each row sums to 1.0 and represents P(class|image)
                       
        Note:
            - Images are automatically converted to grayscale and resized to 28x28
            - MNIST normalization (mean=0.1307, std=0.3081) is applied
            - Returns regular probabilities (via softmax) rather than log probabilities
        """
        # Define preprocessing pipeline matching training data
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),    # Ensure single channel
            transforms.Resize((28, 28)),                    # Standardize size
            transforms.ToTensor(),                          # Convert to [0,1] range
            transforms.Normalize((0.1307,), (0.3081,))      # MNIST dataset statistics
        ])

        # Convert numpy arrays to PIL Images, then to tensors
        tensors = []
        for img in images:
            pil_img = Image.fromarray(img.astype(np.uint8))
            tensor = transform(pil_img)
            tensors.append(tensor)
        
        # Stack into batch tensor
        batch = torch.stack(tensors, dim=0)  # Shape: (num_images, 1, 28, 28)

        # Forward pass without gradient computation (inference mode)
        with torch.no_grad():
            log_probs = self.forward(batch)
            probs = F.softmax(log_probs, dim=1)  # Convert log probabilities to probabilities

        return probs.cpu().numpy()