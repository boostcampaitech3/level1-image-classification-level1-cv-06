#
# F1 Loss (Modified)
# https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
#

import torch
import torch.nn as nn
import torch.nn.functional as F

class F1Loss(nn.Module):
    def __init__(self, classes: int = 18, epsilon: float = 1e-7) -> None:
        super().__init__()

        self.classes = classes
        self.epsilon = epsilon

    def forward(self, y_pred, y_true) -> float:
        assert y_pred.ndim == 2
        assert y_true.ndim == 1

        y_pred = F.softmax(y_pred, dim=1)
        y_true = F.one_hot(y_true, self.classes).to(torch.float32)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        # tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)

        return 1 - f1.mean()