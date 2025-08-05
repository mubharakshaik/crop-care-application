# utils/model.py

import torch.nn as nn
import torchvision.models as models

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNet9, self).__init__()
        base_model = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_model.fc.in_features, num_classes)
        )

    def forward(self, xb):
        xb = self.features(xb)
        xb = self.classifier(xb)
        return xb
