import torch.nn as nn
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights

def build_model():
    model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    return model
