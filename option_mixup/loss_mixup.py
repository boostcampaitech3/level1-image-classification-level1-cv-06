import torch
import torch.nn as nn
import torch.nn.functional as F

'''수정된 부분 (1) 시작'''
# https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
class F1Loss(nn.Module):
    def __init__(self, classes: int = 18, epsilon: float = 1e-7) -> None:
        super().__init__()

        self.classes = classes
        self.epsilon = epsilon

    def forward(self, y_pred, y_true) -> float:
        #print(y_pred.shape) #softmax, shape [64,18]
        #print(y_true.shape) #one-hot vector, shape [64,18]
             
        y_pred = F.softmax(y_pred, dim=1)
        
        tp = torch.sum(y_true * y_pred, dim=-1).to(torch.float32)
        fp = torch.sum((1-y_true) * y_pred, dim=-1).to(torch.float32)
        fn = torch.sum(y_true * (1-y_pred), dim=-1).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)

        return 1 - f1.mean()
'''수정된 부분 (1) 끝'''

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=18, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


_criterion_entrypoints = {
    'cross_entropy': nn.CrossEntropyLoss,
    
    'label_smoothing': LabelSmoothingLoss,
    'f1': F1Loss
}


def criterion_entrypoint(criterion_name):
    return _criterion_entrypoints[criterion_name]


def is_criterion(criterion_name):
    return criterion_name in _criterion_entrypoints


def create_criterion(criterion_name, **kwargs):
    if is_criterion(criterion_name):
        create_fn = criterion_entrypoint(criterion_name)
        criterion = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown loss (%s)' % criterion_name)
    return criterion
