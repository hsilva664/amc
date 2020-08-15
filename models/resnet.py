import torch.nn as nn
import math
from torch.nn import init
import torch.nn.functional as F
import functools

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        def Conv(in_channels, out_channels, kernel_size, padding=1, stride=1,):
            return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        Linear = nn.Linear

        self.conv_0 = Conv(3, 16, 3, 1, 1)
        self.bn_0 = nn.BatchNorm2d(16)

        self.conv1_1a = Conv(16, 16, 3, 1, 1)
        self.bn1_1a = nn.BatchNorm2d(16)
        self.conv1_1b = Conv(16, 16, 3, 1, 1)
        self.bn1_1b = nn.BatchNorm2d(16)
        
        self.conv1_2a = Conv(16, 16, 3, 1, 1)
        self.bn1_2a = nn.BatchNorm2d(16)
        self.conv1_2b = Conv(16, 16, 3, 1, 1)
        self.bn1_2b = nn.BatchNorm2d(16)
        
        self.conv1_3a = Conv(16, 16, 3, 1, 1)
        self.bn1_3a = nn.BatchNorm2d(16)
        self.conv1_3b = Conv(16, 16, 3, 1, 1)
        self.bn1_3b = nn.BatchNorm2d(16)
        
        #
        self.conv2_skip = nn.Conv2d(16, 32, 1, 2, 0, bias=False)

        self.conv2_1a = Conv(16, 32, 3, 1, 2)
        self.bn2_1a = nn.BatchNorm2d(32)
        self.conv2_1b = Conv(32, 32, 3, 1, 1)
        self.bn2_1b = nn.BatchNorm2d(32)
        
        self.conv2_2a = Conv(32, 32, 3, 1, 1)
        self.bn2_2a = nn.BatchNorm2d(32)
        self.conv2_2b = Conv(32, 32, 3, 1, 1)
        self.bn2_2b = nn.BatchNorm2d(32)
        
        self.conv2_3a = Conv(32, 32, 3, 1, 1)
        self.bn2_3a = nn.BatchNorm2d(32)
        self.conv2_3b = Conv(32, 32, 3, 1, 1)
        self.bn2_3b = nn.BatchNorm2d(32)
        
        #
        self.conv3_skip = nn.Conv2d(32, 64, 1, 2, 0, bias=False)
        
        self.conv3_1a = Conv(32, 64, 3, 1, 2)
        self.bn3_1a = nn.BatchNorm2d(64)
        self.conv3_1b = Conv(64, 64, 3, 1, 1)
        self.bn3_1b = nn.BatchNorm2d(64)
        
        self.conv3_2a = Conv(64, 64, 3, 1, 1)
        self.bn3_2a = nn.BatchNorm2d(64)
        self.conv3_2b = Conv(64, 64, 3, 1, 1)
        self.bn3_2b = nn.BatchNorm2d(64)
        
        self.conv3_3a = Conv(64, 64, 3, 1, 1)
        self.bn3_3a = nn.BatchNorm2d(64)
        self.conv3_3b = Conv(64, 64, 3, 1, 1)
        self.bn3_3b = nn.BatchNorm2d(64)
        
        #
        self.prunable = [self.conv_0, self.conv1_1a, self.conv1_1b, self.conv1_2a, self.conv1_2b, self.conv1_3a, self.conv1_3b, self.conv2_1a, self.conv2_1b, self.conv2_2a, self.conv2_2b, self.conv2_3a, self.conv2_3b, self.conv3_1a, self.conv3_1b, self.conv3_2a, self.conv3_2b, self.conv3_3a, self.conv3_3b]        
        
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(64, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv_0(x)
        x = F.relu(self.bn_0(x), inplace=True)
        #
        residual = x
        x = self.conv1_1a(x)
        x = F.relu(self.bn1_1a(x), inplace=True)
        x = self.conv1_1b(x)
        x = F.relu(residual + self.bn1_1b(x), inplace=True)
        
        residual = x
        x = self.conv1_2a(x)
        x = F.relu(self.bn1_2a(x), inplace=True)
        x = self.conv1_2b(x)
        x = F.relu(residual + self.bn1_2b(x), inplace=True)
        
        residual = x
        x = self.conv1_3a(x)
        x = F.relu(self.bn1_3a(x), inplace=True)
        x = self.conv1_3b(x)
        x = F.relu(residual + self.bn1_3b(x), inplace=True)
        #
        residual = self.conv2_skip(x)
        x = self.conv2_1a(x)
        x = F.relu(self.bn2_1a(x), inplace=True)
        x = self.conv2_1b(x)
        x = F.relu(residual + self.bn2_1b(x), inplace=True)
        
        residual = x
        x = self.conv2_2a(x)
        x = F.relu(self.bn2_2a(x), inplace=True)
        x = self.conv2_2b(x)
        x = F.relu(residual + self.bn2_2b(x), inplace=True)
        
        residual = x
        x = self.conv2_3a(x)
        x = F.relu(self.bn2_3a(x), inplace=True)
        x = self.conv2_3b(x)
        x = F.relu(residual + self.bn2_3b(x), inplace=True)
        #
        residual = self.conv3_skip(x)
        x = self.conv3_1a(x)
        x = F.relu(self.bn3_1a(x), inplace=True)
        x = self.conv3_1b(x)
        x = F.relu(residual + self.bn3_1b(x), inplace=True)
        
        residual = x
        x = self.conv3_2a(x)
        x = F.relu(self.bn3_2a(x), inplace=True)
        x = self.conv3_2b(x)
        x = F.relu(residual + self.bn3_2b(x), inplace=True)
        
        residual = x
        x = self.conv3_3a(x)
        x = F.relu(self.bn3_3a(x), inplace=True)
        x = self.conv3_3b(x)
        x = F.relu(residual + self.bn3_3b(x), inplace=True)
        #
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x