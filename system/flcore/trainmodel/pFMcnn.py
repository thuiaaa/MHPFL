import math
import torch
import torchvision

from torch import nn
from torch.nn import functional as F


"""
The 5 different CNN models in pFedMoE
"""

class pFMCNN_1(nn.Module):
    #Conv1: 5x5, 16            after conv: (H - kernel_size + 2*padding)/stride + 1
    #Conv2: 5x5, 32
    #FC1: 2000
    #FC2: 500
    #FC3: 10/100
    def __init__(self, in_channels=3, num_classes=10, dim=800):
        super().__init__()
        self.conv1 = nn.Sequential(                      #input [3,32,32]
            nn.Conv2d(in_channels, 16, kernel_size=5),   #-> [16,28,28]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             #-> [16,14,14]
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 2000),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2000, 500),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Linear(500, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.head(out)
        return out

    def forward_nohead(self, x):
        out = self.conv1(x)
        out = self.conv2(x)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class pFMCNN_2(nn.Module):
    #Conv1: 5x5, 16            after conv: (H - kernel_size + 2*padding)/stride + 1
    #Conv2: 5x5, 16
    #FC1: 2000
    #FC2: 500
    #FC3: 10/100
    def __init__(self, in_channels=3, num_classes=10, dim=400):
        super().__init__()
        self.conv1 = nn.Sequential(                      #input [3,32,32]
            nn.Conv2d(in_channels, 16, kernel_size=5),   #-> [16,28,28]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             #-> [16,14,14]
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5),            #-> [16,10,10]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)                              #-> [16,5,5]
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 2000),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2000, 500),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Linear(500, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.head(out)
        return out

    def forward_nohead(self, x):
        out = self.conv1(x)
        out = self.conv2(x)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class pFMCNN_3(nn.Module):
    #Conv1: 5x5, 16            after conv: (H - kernel_size + 2*padding)/stride + 1
    #Conv2: 5x5, 32
    #FC1: 1000
    #FC2: 500
    #FC3: 10/100
    def __init__(self, in_channels=3, num_classes=10, dim=800):
        super().__init__()
        self.conv1 = nn.Sequential(                      #input [3,32,32]
            nn.Conv2d(in_channels, 16, kernel_size=5),   #-> [16,28,28]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             #-> [16,14,14]
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5),            #-> [32,10,10]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)                              #-> [32,5,5]
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 1000),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Linear(500, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.head(out)
        return out

    def forward_nohead(self, x):
        out = self.conv1(x)
        out = self.conv2(x)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class pFMCNN_4(nn.Module):
    #Conv1: 5x5, 16            after conv: (H - kernel_size + 2*padding)/stride + 1
    #Conv2: 5x5, 32
    #FC1: 800
    #FC2: 500
    #FC3: 10/100
    def __init__(self, in_channels=3, num_classes=10, dim=800):
        super().__init__()
        self.conv1 = nn.Sequential(                      #input [3,32,32]
            nn.Conv2d(in_channels, 16, kernel_size=5),   #-> [16,28,28]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             #-> [16,14,14]
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5),            #-> [32,10,10]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)                              #-> [32,5,5]
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 800),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(800, 500),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Linear(500, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.head(out)
        return out

    def forward_nohead(self, x):
        out = self.conv1(x)
        out = self.conv2(x)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class pFMCNN_5(nn.Module):
    #Conv1: 5x5, 16            after conv: (H - kernel_size + 2*padding)/stride + 1
    #Conv2: 5x5, 32
    #FC1: 500
    #FC2: 500
    #FC3: 10/100
    def __init__(self, in_channels=3, num_classes=10, dim=800):
        super().__init__()
        self.conv1 = nn.Sequential(                      #input [3,32,32]
            nn.Conv2d(in_channels, 16, kernel_size=5),   #-> [16,28,28]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             #-> [16,14,14]
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5),            #-> [32,10,10]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)                              #-> [32,5,5]
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 500),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(500, 500),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Linear(500, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.head(out)
        return out

    def forward_nohead(self, x):
        out = self.conv1(x)
        out = self.conv2(x)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out