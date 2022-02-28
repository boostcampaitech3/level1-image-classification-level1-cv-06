#
# MultiHeadClassifier
#

import torch
import torch.nn as nn
import torchvision.models as models

class MultiHeadClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.resnet34 = models.resnet34(pretrained=True)
        self.resnet34.fc = nn.Linear(512, 18, bias=True)
        torch.nn.init.kaiming_normal_(self.resnet34.fc.weight)
        torch.nn.init.zeros_(self.resnet34.fc.bias)

        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Linear(2048, 18, bias=True)
        torch.nn.init.kaiming_normal_(self.resnet50.fc.weight)
        torch.nn.init.zeros_(self.resnet50.fc.bias)

        self.resnet101 = models.resnet101(pretrained=True)
        self.resnet101.fc = nn.Linear(2048, 18, bias=True)
        torch.nn.init.kaiming_normal_(self.resnet101.fc.weight)
        torch.nn.init.zeros_(self.resnet101.fc.bias)

        self.resnet152 = models.resnet152(pretrained=True)
        self.resnet152.fc = nn.Linear(2048, 18, bias=True)
        torch.nn.init.kaiming_normal_(self.resnet152.fc.weight)
        torch.nn.init.zeros_(self.resnet152.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ResNet Only Ensemble Test
        results = [self.resnet34(x), self.resnet50(x), self.resnet101(x), self.resnet152(x)]
        return torch.mean(torch.stack(results, dim=0), dim=0)
