#
# MultiHeadClassifier
#

import torch
import torch.nn as nn
import torchvision.models as models

class MultiHeadClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.fc = nn.Identity()
        for param in self.resnet18.parameters():
            param.requires_grad = False

        # self.fc_mask = nn.Linear(512, 3, bias=True)
        # self.fc_gender = nn.Linear(512, 2, bias=True)
        # self.fc_age = nn.Linear(512, 3, bias=True)

        # torch.nn.init.kaiming_normal_(self.fc_mask.weight)
        # torch.nn.init.kaiming_normal_(self.fc_gender.weight)
        # torch.nn.init.kaiming_normal_(self.fc_age.weight)
        # torch.nn.init.zeros_(self.fc_mask.bias)
        # torch.nn.init.zeros_(self.fc_gender.bias)
        # torch.nn.init.zeros_(self.fc_age.bias)

        self.fc = nn.Linear(512, 18, bias=True)
        torch.nn.init.kaiming_normal_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet18(x)
        return self.fc(x)
