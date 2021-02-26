import torch
from torch import nn
from torchvision.models import resnet50
from efficientnet_pytorch import EfficientNet


"""
Here every model to be used for pretraining/training is defined.
"""


class PlainResnet50(nn.Module):
    def __init__(self):
        super(PlainResnet50, self).__init__()
        
        base_model = resnet50()
        self.block = nn.Sequential(
            base_model,
            nn.Linear(1000, 26),
        )
        
        nn.init.xavier_normal_(self.block[1].weight)
        
    def forward(self, x):
        out = self.block(x)
        return out


class CustomResnet50(nn.Module):
    def __init__(self):
        super(CustomResnet50, self).__init__()
        
        base_model = resnet50()
        base_model.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.BatchNorm1d(128),
            nn.Linear(128, 26),
        )
        
        self.block = nn.Sequential(
            base_model
        )
        
        nn.init.xavier_normal_(self.block[0].fc[0].weight)
        nn.init.xavier_normal_(self.block[0].fc[-1].weight)
    
    def forward(self, x):
        out = self.block(x)
        return out


class PlainEfficientnetB4(nn.Module):
    def __init__(self):
        super(PlainEfficientnetB4, self).__init__()
        
        base_model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=26)
        self.block = nn.Sequential(
            base_model
        )
        
        nn.init.xavier_normal_(self.block[0]._fc.weight)
        
    def forward(self, x):
        out = self.block(x)
        return out


class PlainEfficientnetB5(nn.Module):
    def __init__(self):
        super(PlainEfficientnetB5, self).__init__()
        
        base_model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=26)
        self.block = nn.Sequential(
            base_model
        )
        
        nn.init.xavier_normal_(self.block[0]._fc.weight)
        
    def forward(self, x):
        out = self.block(x)
        return out


class PlainEfficientnetB7(nn.Module):
    def __init__(self):
        super(PlainEfficientnetB7, self).__init__()
        
        base_model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=26)
        self.block = nn.Sequential(
            base_model
        )
        
        nn.init.xavier_normal_(self.block[0]._fc.weight)
        
    def forward(self, x):
        out = self.block(x)
        return out
    

# class PretrainedResnet(nn.Module):
    
#     def __init__(self, resnet_base):
#         super(PretrainedResnet, self).__init__()

#         self.block = nn.Sequential(
#             #nn.Conv2d(1, 3, 1, stride=1),
#             #nn.ReLU(),
#             resnet_base,
#         )

#     def forward(self, x):
#         out = self.block(x)
#         return out