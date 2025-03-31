from __future__ import print_function  # Ensures compatibility of print() in Python 2.x
import torch
import torch.nn as nn  # Provides neural network layers
import torch.nn.functional as F  # Provides activation functions and other operations
import torch.optim as optim  # Provides optimization algorithms
from torchvision import datasets, transforms  # For loading datasets and applying transformations
from torch.optim.lr_scheduler import StepLR  # For scheduling the learning rate


MEAN = 0.1307
STD = 0.3081
# -------------------------------
# Define the Neural Network Model
# -------------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First convolutional layer: takes a 1-channel (grayscale) image, produces 32 channels
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # Second convolutional layer: takes 32 channels, produces 64 channels
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # Dropout layers help prevent overfitting by randomly zeroing out some activations
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        # Fully connected layer: input size 9216 (from the flattened feature maps) to 128 features
        self.fc1 = nn.Linear(9216, 128)
        # Output layer: maps 128 features to 10 output classes (digits 0-9)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        
        #Reshape
        x = x.reshape(280, 280, 4)
        #x = x [:, :, 3]
        x = torch.narrow(x, dim=2, start=3, length=1)
        x = x.reshape(1, 1, 280, 280)
        x = F.avg_pool2d(x, 10, stride=10)
        x = x/255
        x = (x - MEAN) / STD


        # First convolution and ReLU activation
        x = self.conv1(x)
        x = F.relu(x)
        # Second convolution
        x = self.conv2(x)
        # Apply max pooling with a 2x2 window to reduce spatial dimensions
        x = F.max_pool2d(x, 2)
        # Apply dropout to reduce overfitting
        x = self.dropout1(x)
        # Flatten the tensor for the fully connected layers
        x = torch.flatten(x, 1)
        # Fully connected layer with ReLU activation
        x = self.fc1(x)
        x = F.relu(x)
        # Second dropout layer
        x = self.dropout2(x)
        # Final output layer
        x = self.fc2(x)
        # Apply log softmax to obtain log-probabilities for each class
        output = F.softmax(x, dim=1)
        return output