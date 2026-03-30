import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

def create_resnet18(pretrained=True, dropout=0.2):
    """
    Create ResNet18 model for age regression
    
    Args:
        pretrained: whether to use pretrained weights
        dropout: dropout rate for the final layer
    
    Returns:
        PyTorch model
    """
    if pretrained:
        weights = ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18()
    
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 1)
    )
    
    return model

def get_device():
    """Get available device (cuda/mps/cpu)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
