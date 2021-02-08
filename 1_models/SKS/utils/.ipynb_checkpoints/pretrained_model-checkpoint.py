import os
import torch
from torch import nn
import torchvision
from torchvision.models import resnet50
from efficientnet_pytorch import EfficientNet


class ModelWrapper(nn.Module):
    def __init__(self, base_model):
        super(ModelWrapper, self).__init__()
        
        self.block = nn.Sequential(
            base_model
        )
        
    def forward(self, x):
        out = self.block(x)
        return out

    
class CustomModel(nn.Module):
    """
    To add custom layers in base model, e.g. sigmoid layer.
    """
    def __init__(self, base_model):
        super(CustomModel, self).__init__()
        self.block = nn.Sequential(
            base_model,
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        out = self.block(x)
        return out


class CallPretrainedModel():
    """
    model_type: [resnet50, efficientnet]
    """
    def __init__(self, mode='train', model_index = None, model_type=None, path='./pretrained_model'):

        self.model_index = model_index
        self.model_type = model_type
        
        if model_type == 'resnet50':
        
            weight_path = os.path.join(path, 'pretrained_resnet.pth')
            base_model = resnet50()
            base_model.fc = nn.Sequential(
                nn.Linear(2048, 256),
                nn.BatchNorm1d(256),
                nn.Linear(256, 26),
            )
            base_model = ModelWrapper(base_model)
            model = CallPretrainedModel._load_weights(base_model, weight_path)
            
        elif model_type == 'efficientnet':
            
            weight_path = os.path.join(path, 'pretrained_efficientnet.pth')
            base_model = EfficientNet()
            model = CallPretrainedModel._load_weights(base_model, weight_path)
        
        else:
            raise Exception(f"No such pretrained model: {model_type}")
        
        self.return_model = model
        self.mediated_model = None
        
        
    def customize(self):
        return_model = CustomModel(self.return_model)
        self.mediated_model = return_model
        return return_model
    
    
    def load_trained_weight(self, model_index=0, model_type='early', trained_weight_path='./ckpt'):
        
        assert model_index > 0
        
        model_name = 'early_stopped.pth' if model_type == 'early' else f'model_ckpt_{model_type}.pth'
        ckpt_path = os.path.join(trained_weight_path, f'model_{model_index}', model_name)
        
        trained_model = self.mediated_model
        trained_model.load_state_dict(torch.load(ckpt_path))
        trained_model.eval()
        
        return trained_model
    
    
    @staticmethod
    def _load_weights(model, path):
        model.load_state_dict(torch.load(path))
        return model
