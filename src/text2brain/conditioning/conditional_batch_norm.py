
import torch
import torch.nn as nn


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, n_features, n_classes, eps=1e-5, momentum=0.1):
        super(ConditionalBatchNorm2d, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        
        self.bn = nn.BatchNorm2d(n_features, affine=False, eps=eps, momentum=momentum)
        
        self.gamma = nn.Embedding(n_classes, n_features)
        self.beta = nn.Embedding(n_classes, n_features)
        
        self.gamma.weight.data.fill_(1)
        self.beta.weight.data.fill_(0)
    
    def forward(self, x, y):
        out = self.bn(x)
        
        gamma = self.gamma(y).unsqueeze(2).unsqueeze(3)
        beta = self.beta(y).unsqueeze(2).unsqueeze(3)
        
        out = gamma * out + beta
        return out