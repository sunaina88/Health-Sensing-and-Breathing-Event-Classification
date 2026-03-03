# in cnn_model.py, we define a simple 1D CNN which classifies breathing windows into Normal, Hypopnea and Apnea
# this is invoked in train_model.py

# first, we import the libraries
import torch
import torch.nn as nn


# this class defines the 1D CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        # this is the first convolution layer with 1 input channel and 16 output channels
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5)
        # here, we downsample by a factor of 2
        self.pool = nn.MaxPool1d(2)
        # this is the second convolution layer
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5)
        # these are the fully connected layers for classification
        self.fc1 = nn.Linear(32 * 237, 64)   # 32*237 because of flattened feature size after convolution and pooling
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # here, there are convolution, activation and pooling
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        # we flatten before fully connected layers
        x = x.view(x.size(0), -1)
        # these are dense layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x