import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from efficientnet_pytorch import EfficientNet


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()

        self.fc = nn.Linear(1536, 18, bias=True)
        #torch.nn.init.kaiming_normal_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)

        self.effnetb3 = EfficientNet.from_pretrained('efficientnet-b3', num_classes=18)
        self.effnetb3.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            self.fc
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.effnetb3(x)
