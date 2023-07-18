import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim=28*28, out_dim=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.classifier = nn.Linear(100, out_dim)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)  

    def forward(self, x):
        b = x.size(0)
        x = x.view(b, -1)
                
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.classifier(x)
        
        return x    

    
class ConvNet(nn.Module):

    def __init__(self, out_dim=10):

        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.classifier = nn.Linear(500, out_dim)

        self.encoder = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            self.conv2,
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            self.fc1,
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        
        return x
    