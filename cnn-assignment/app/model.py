# app/model.py
import torch.nn as nn

class SpecCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*16*16, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


