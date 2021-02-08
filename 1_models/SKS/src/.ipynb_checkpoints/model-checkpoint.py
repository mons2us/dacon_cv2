import torch
from torch import nn
from torchvision.models import resnet50

class DaconModel(nn.Module):
    
    def __init__(self) -> None:
        
        super().__init__()
        
        self.resnet = resnet50(pretrained=True)
        self.classifier = nn.Linear(1000, 26)

    def forward(self, x):
        out = self.resnet(x)
        out = self.classifier(out)

        return out
    

class PretrainedResnet(nn.Module):
    
    def __init__(self, resnet_base):
        super(PretrainedResnet, self).__init__()

        self.block = nn.Sequential(
            #nn.Conv2d(1, 3, 1, stride=1),
            #nn.ReLU(),
            resnet_base,
        )

    def forward(self, x):
        out = self.block(x)
        return out