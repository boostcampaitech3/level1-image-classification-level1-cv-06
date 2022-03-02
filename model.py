#
# boostcamp AI Tech
# Image Classification Competition
#

import torch
import torch.nn as nn
import torchvision.models as models

class ExperimentalClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fc = nn.Linear(1536, 18, bias=True)
        torch.nn.init.kaiming_normal_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)

        self.effnetb3 = models.efficientnet_b3(pretrained=True)
        self.effnetb3.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            self.fc
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.effnetb3(x)
