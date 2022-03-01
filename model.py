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

        self.fc = nn.Linear(1792, 18, bias=True)
        torch.nn.init.kaiming_normal_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)

        self.effnetb4 = models.efficientnet_b4(pretrained=True)
        self.effnetb4.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            self.fc
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.effnetb4(x)
