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

        self.fc_mask = nn.Linear(512, 3, bias=True)
        self.fc_gender = nn.Linear(512, 2, bias=True)
        self.fc_age = nn.Linear(512, 3, bias=True)

        torch.nn.init.kaiming_normal_(self.fc_mask.weight)
        torch.nn.init.kaiming_normal_(self.fc_gender.weight)
        torch.nn.init.kaiming_normal_(self.fc_age.weight)
        torch.nn.init.zeros_(self.fc_mask.bias)
        torch.nn.init.zeros_(self.fc_gender.bias)
        torch.nn.init.zeros_(self.fc_age.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet18(x)

        mask = self.fc_mask(x)
        gender = self.fc_gender(x)
        age = self.fc_age(x)

        mask_gender = torch.einsum("ij,ik->ijk", (mask, gender)).reshape(mask.shape[0], -1)
        mask_gender_age = torch.einsum("ij,ik->ijk", (mask_gender, age)).reshape(mask_gender.shape[0], -1)

        return mask_gender_age
    
class ResNet18_model(nn.Module):
    def __init__(self,num_classes):
        super(ResNet18_model, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)
        torch.nn.init.kaiming_normal_(self.backbone.fc.weight)
        torch.nn.init.zeros_(self.backbone.fc.bias)
    def forward(self,x):
        x = self.backbone(x)
        return x
class ResNet34_model(nn.Module):
    def __init__(self,num_classes):
        super(ResNet34_model, self).__init__()
        self.backbone = models.resnet34(pretrained=True)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)
        torch.nn.init.kaiming_normal_(self.backbone.fc.weight)
        torch.nn.init.zeros_(self.backbone.fc.bias)
    def forward(self,x):
        x = self.backbone(x)
        return x
class ResNet50_model(nn.Module):
    def __init__(self,num_classes):
        super(ResNet50_model, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)
        torch.nn.init.kaiming_normal_(self.backbone.fc.weight)
        torch.nn.init.zeros_(self.backbone.fc.bias)
    def forward(self,x):
        x = self.backbone(x)
        return x
class ResNet101_model(nn.Module):
    def __init__(self,num_classes):
        super(ResNet101_model, self).__init__()
        self.backbone = models.resnet101(pretrained=True)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)
        torch.nn.init.kaiming_normal_(self.backbone.fc.weight)
        torch.nn.init.zeros_(self.backbone.fc.bias)
    def forward(self,x):
        x = self.backbone(x)
        return x
class ResNet152_model(nn.Module):
    def __init__(self,num_classes):
        super(ResNet152_model, self).__init__()
        self.backbone = models.resnet34(pretrained=True)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)
        torch.nn.init.kaiming_normal_(self.backbone.fc.weight)
        torch.nn.init.zeros_(self.backbone.fc.bias)
    def forward(self,x):
        x = self.backbone(x)
        return x
class ResNet_Ensemble(nn.Module):
    def __init__(self):
        super(ResNet_Ensemble, self).__init__()
        num_classes = 18
        self.model1 = ResNet18_model(num_classes)
        self.model2 = ResNet34_model(num_classes)
        self.model3 = ResNet50_model(num_classes)
        self.model4 = ResNet101_model(num_classes)
        self.model5 = ResNet152_model(num_classes)
        self.integrator = nn.Linear(num_classes*5,num_classes,bias=False)
        torch.nn.init.kaiming_normal_(self.integrator.weight)
        self.Sigmoid = nn.Sigmoid()
        #loss = 0.2*1*loss1 + 0.2*0.8*loss2 + 0.2*0.6*loss3 + 0.2*0.4*loss4 + 0.2*0.2*loss5 + 0.8*losses
    def forward(self,x):
        y1 = self.model1(x)
        y2 = self.model2(x)
        y3 = self.model3(x)
        y4 = self.model4(x)
        y5 = self.model5(x)
        y = torch.cat((y1,y2,y3,y4,y5),1)
        y = self.integrator(y)
        y = self.Sigmoid(y)
        return y1,y2,y3,y4,y5,y

class ResNet_Ensemble2_original(nn.Module):
    def __init__(self):
        super(ResNet_Ensemble2_original, self).__init__()
        num_classes = 18
        self.model1 = ResNet18_model(num_classes)
        self.model2 = ResNet34_model(num_classes)
        self.model3 = ResNet50_model(num_classes)
        self.model4 = ResNet101_model(num_classes)
        self.model5 = ResNet152_model(num_classes)
        self.integrator = nn.Linear(num_classes*5,num_classes,bias=False)
        torch.nn.init.kaiming_normal_(self.integrator.weight)
        
        
    def forward(self,x):
        y1 = self.model1(x)
        y1 = torch.mul(y1,0.3*1)
        y2 = self.model2(x)
        y2 = torch.mul(y2,0.3*0.8)
        y3 = self.model3(x)
        y3 = torch.mul(y3,0.3*0.6)
        y4 = self.model4(x)
        y4 = torch.mul(y4,0.3*0.4)
        y5 = self.model5(x)
        y5 = torch.mul(y5,0.3*0.2)
        y = torch.cat((y1,y2,y3,y4,y5),1)
        y = self.integrator(y)
        y = torch.mul(y,0.1)
        return y1,y2,y3,y4,y5,y
    
class ResNet_Ensemble2(nn.Module):
    def __init__(self):
        super(ResNet_Ensemble2, self).__init__()
        num_classes = 18
        self.model1 = ResNet18_model(num_classes)
        self.model2 = ResNet34_model(num_classes)
        self.model3 = ResNet50_model(num_classes)
        self.model4 = ResNet101_model(num_classes)
        self.model5 = ResNet152_model(num_classes)
        self.integrator = nn.Linear(num_classes*5,num_classes,bias=False)
        torch.nn.init.kaiming_normal_(self.integrator.weight)
        self.Sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        y1 = self.model1(x)
        #y1 = self.Sigmoid(y1)
        y1 = torch.mul(y1,0.3*1)
        y2 = self.model2(x)
        #y2 = self.Sigmoid(y2)
        y2 = torch.mul(y2,0.3*0.8)
        y3 = self.model3(x)
        #y3 = self.Sigmoid(y3)
        y3 = torch.mul(y3,0.3*0.6)
        y4 = self.model4(x)
        #y4 = self.Sigmoid(y4)
        y4 = torch.mul(y4,0.3*0.4)
        y5 = self.model5(x)
        #y5 = self.Sigmoid(y5)
        y5 = torch.mul(y5,0.3*0.2)
        y = torch.cat((y1,y2,y3,y4,y5),1)
        y = self.integrator(y)
        y = self.Sigmoid(y)
        y = torch.mul(y,0.5)
        return y1,y2,y3,y4,y5,y
    
class ResNet_Ensemble_debugged(nn.Module):
    def __init__(self):
        super(ResNet_Ensemble_debugged, self).__init__()
        num_classes = 18
        self.model1 = ResNet18_model(num_classes)
        self.model2 = ResNet34_model(num_classes)
        self.model3 = ResNet50_model(num_classes)
        self.model4 = ResNet101_model(num_classes)
        self.model5 = ResNet152_model(num_classes)
        self.integrator = nn.Linear(num_classes*5,num_classes,bias=False)
        self.refiner = nn.Linear(num_classes,num_classes)
        torch.nn.init.kaiming_normal_(self.integrator.weight)
        torch.nn.init.kaiming_normal_(self.refiner.weight)
        self.Tanh = nn.Tanh()
        self.Sigmoid = nn.Sigmoid()
    def forward(self,x):
        # Parallel Module part
        y1 = self.model1(x)
        #y1 = self.refiner(y1)
        y1 = self.Tanh(y1)
        
        y2 = self.model2(x)
        #y2 = self.refiner(y2)
        y2 = self.Tanh(y2)
        
        y3 = self.model3(x)
        #y3 = self.refiner(y3)
        y3 = self.Tanh(y3)
        
        y4 = self.model4(x)
        #y4 = self.refiner(y4)
        y4 = self.Tanh(y4)
        
        y5 = self.model5(x)
        #y5 = self.refiner(y5)
        y5 = self.Tanh(y5)
        
        y6 = torch.cat((y1,y2,y3,y4,y5),1)
        y6 = self.integrator(y6)
        y6 = self.Sigmoid(y6)
        
        # Weighted Summation part
        #y = torch.mul(y6,0.1)
        #y1 = torch.mul(y1,0.3*1)
        #y2 = torch.mul(y2,0.3*0.8)
        #y3 = torch.mul(y3,0.3*0.6)
        #y4 = torch.mul(y4,0.3*0.4)
        #y5 = torch.mul(y5,0.3*0.2)
        
        # Final output part
        #y += y1+y2+y3+y4+y5
        y = y1+y2+y3+y4+y5+y6
        return y1,y2,y3,y4,y5,y6,y
    
    
class ResNet_Ensemble_final1(nn.Module):
    def __init__(self):
        super(ResNet_Ensemble_final1, self).__init__()
        num_classes = 18
        self.model1 = ResNet18_model(num_classes)
        self.model2 = ResNet34_model(num_classes)
        self.model3 = ResNet50_model(num_classes)
        self.model4 = ResNet101_model(num_classes)
        self.model5 = ResNet152_model(num_classes)
        self.integrator = nn.Linear(num_classes*5,num_classes,bias=False)
        torch.nn.init.kaiming_normal_(self.integrator.weight)
        self.Tanh = nn.Tanh()
        self.Sigmoid = nn.Sigmoid()
    def forward(self,x):
        # Parallel Module part
        y1 = self.model1(x)
        y1 = self.Tanh(y1)
        y2 = self.model2(x)
        y2 = self.Tanh(y2)
        y3 = self.model3(x)
        y3 = self.Tanh(y3)
        y4 = self.model4(x)
        y4 = self.Tanh(y4)
        y5 = self.model5(x)
        y5 = self.Tanh(y5)
        y6 = torch.cat((y1,y2,y3,y4,y5),1)
        y6 = self.integrator(y6)
        y6 = self.Tanh(y6)
        # Weighted Summation part
        y = torch.mul(y6,0.1)
        y1 = torch.mul(y1,0.3*1)
        y2 = torch.mul(y2,0.3*0.8)
        y3 = torch.mul(y3,0.3*0.6)
        y4 = torch.mul(y4,0.3*0.4)
        y5 = torch.mul(y5,0.3*0.2)
        # Final output part
        y += y1+y2+y3+y4+y5
        return y1,y2,y3,y4,y5,y6,y