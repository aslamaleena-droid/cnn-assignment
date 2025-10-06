import torch, torch.nn as nn, torch.nn.functional as F
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.pool  = nn.MaxPool2d(2,2)
        self.fc1   = nn.Linear(32*7*7,64)
        self.fc2   = nn.Linear(64,num_classes)
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))  # 28->14
        x = self.pool(F.relu(self.conv2(x)))  # 14->7
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
