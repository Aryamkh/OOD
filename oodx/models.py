import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math

class CosineClassifier(nn.Module):
    def __init__(self, in_dim, num_classes, scale=30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.scale = nn.Parameter(torch.tensor(float(scale)), requires_grad=False)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        return self.scale * (x @ w.t())

class BackboneWithHead(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()
        name = args.backbone
        if name == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.feat_dim = net.fc.in_features
            self.backbone = nn.Sequential(*list(net.children())[:-1], nn.Flatten())
        elif name == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.feat_dim = net.fc.in_features
            self.backbone = nn.Sequential(*list(net.children())[:-1], nn.Flatten())
        elif name == "convnext_tiny":
            net = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
            self.feat_dim = net.classifier[2].in_features
            self.backbone = nn.Sequential(
                net.features, nn.AdaptiveAvgPool2d(1), nn.Flatten()
            )
        else:
            raise NotImplementedError(name)

        if getattr(args, "cosine_head", 0):
            self.head = CosineClassifier(self.feat_dim, num_classes, scale=args.cosine_scale)
        else:
            self.head = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x, return_feat=False):
        feat = self.backbone(x)
        if feat.ndim > 2:
            feat = feat.flatten(1)
        logits = self.head(feat)
        return (logits, feat) if return_feat else logits
