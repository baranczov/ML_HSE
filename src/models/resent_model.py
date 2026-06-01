import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights


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

def create_resnet50(pretrained=True, dropout=0.2):
    """ResNet50 for age regression — used as teacher in knowledge distillation."""
    if pretrained:
        weights = ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    else:
        model = models.resnet50()

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 1)
    )
    return model


def get_embedding_dim(model):
    """Return the output dim of the backbone (before the fc head)."""
    # Works for any ResNet: fc is Sequential(Dropout, Linear(in, 1))
    return model.fc[1].in_features


def create_model(name, pretrained=True, dropout=0.2):
    """Factory: 'resnet18' or 'resnet50'."""
    if name == "resnet18":
        return create_resnet18(pretrained=pretrained, dropout=dropout)
    if name == "resnet50":
        return create_resnet50(pretrained=pretrained, dropout=dropout)
    raise ValueError(f"Unknown model name: {name}")


def get_device():
    """Get available device (cuda/mps/cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
