import torch
import torch.nn as nn
import torch.nn.functional as F

class DualInputConvNet(nn.Module):
    def __init__(self):
        super(DualInputConvNet, self).__init__()
        # Define the convolutional layers
        self.num_flat_features = 5 * 64 * 64  # Change 64 to your actual spatial dimensions

        # Fully connected layers
        self.fc1 = nn.Linear(10, 16)  # First fully connected layer
        self.fc2 = nn.Linear(16, 32)  # Second fully connected layer
        self.fc3 = nn.Linear(32, 5)  #
         # Reduce back to 10 channels

    def forward(self, x1, x2):
        # print("x1, x2 shape: " , x1.shape, x2.shape)
        # Concatenate the two inputs along the channel axis (dimension 1)
        x = torch.cat((x1, x2), dim=1)  # Concatenates along channels, resulting in (B, 10, 32, 64) input tensor
        B, a,b,c = x.shape
        # print("x.shape: " , x.shape)
        # Apply convolutional layers with ReLU activations
        x = x.reshape(-1, a)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) 
        x = x.reshape(B,-1,b,c)
        # print("x.shape: " , x.shape)
        return x
