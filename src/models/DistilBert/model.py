from typing import Dict, Optional, Tuple
import src.config as config
import src.trainer as trainer

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

import sys

class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 24, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(24, 48, 5)
        self.fc1 = nn.Linear(1728, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

def model_init(conf: config.Config, dataset: data.Dataset) -> Tuple[trainer.Trainer, Optional[Dict]]:
    print("hello from the dummy model")
    sys.exit(0)
    model = Dummy() 
    loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size.value, shuffle=True, num_workers=2)
    return None, None