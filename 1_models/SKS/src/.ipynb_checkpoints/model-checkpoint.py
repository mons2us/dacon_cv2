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