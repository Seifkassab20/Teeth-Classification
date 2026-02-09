import torch.nn as nn
from torchvision import models

class PretrainedModel(nn.Module):
    
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet50(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes
        )

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)
