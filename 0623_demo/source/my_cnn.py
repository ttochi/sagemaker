import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(50, 500, 4, stride=1, padding=0)        
        self.fc1 = torch.nn.Conv2d(500, 10, 1, stride=1, padding=0)        
    
    def forward(self, img):
        x = self.conv1(img)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        
        x = self.fc1(x)
        
        return torch.nn.functional.log_softmax(x.squeeze(), dim=1)