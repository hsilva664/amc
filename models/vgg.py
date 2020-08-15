import torch.nn as nn
import math
from torch.nn import init
import torch.nn.functional as F

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        in_channels = 3
        
        Conv = nn.Conv2d
        Linear = nn.Linear
        
        self.pool = nn.MaxPool2d(2)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(True)
        
        self.conv1 = Conv(in_channels, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = Conv(64, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = Conv(64, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = Conv(128, 128, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.conv5 = Conv(128, 256, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = Conv(256, 256, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = Conv(256, 256, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = Conv(256, 256, 3, 1, 1)
        self.bn8 = nn.BatchNorm2d(256)
        
        self.conv9 = Conv(256, 512, 3, 1, 1)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = Conv(512, 512, 3, 1, 1)
        self.bn10 = nn.BatchNorm2d(512)
        self.conv11 = Conv(512, 512, 3, 1, 1)
        self.bn11 = nn.BatchNorm2d(512)
        self.conv12 = Conv(512, 512, 3, 1, 1)
        self.bn12 = nn.BatchNorm2d(512)
        
        self.conv13 = Conv(512, 512, 3, 1, 1)
        self.bn13 = nn.BatchNorm2d(512)
        self.conv14 = Conv(512, 512, 3, 1, 1)
        self.bn14 = nn.BatchNorm2d(512)
        self.conv15 = Conv(512, 512, 3, 1, 1)
        self.bn15 = nn.BatchNorm2d(512)
        self.conv16 = Conv(512, 512, 3, 1, 1)
        self.bn16 = nn.BatchNorm2d(512)
        
        self.linear1 = nn.Linear(512, 10)

        self.prunable = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8, self.conv9, self.conv10, self.conv11, self.conv12, self.conv13, self.conv14, self.conv15, self.conv16]

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
        x = self.conv1(x)
        
        x = self.conv2(self.relu(self.bn1(x)))
        
        
        x = self.pool(self.relu(self.bn2(x)))
        
        x = self.conv3(x)
        
        x = self.conv4(self.relu(self.bn3(x)))
        
        
        x = self.pool(self.relu(self.bn4(x)))
        
        x = self.conv5(x)
        
        x = self.conv6(self.relu(self.bn5(x)))
        
        x = self.conv7(self.relu(self.bn6(x)))
        
        x = self.conv8(self.relu(self.bn7(x)))
        
        
        x = self.pool(self.relu(self.bn8(x)))
        
        x = self.conv9(x)
        
        x = self.conv10(self.relu(self.bn9(x)))
        
        x = self.conv11(self.relu(self.bn10(x)))
        
        x = self.conv12(self.relu(self.bn11(x)))
        
        
        x = self.pool(self.relu(self.bn12(x)))
        
        x = self.conv13(x)
        
        x = self.conv14(self.relu(self.bn13(x)))
        
        x = self.conv15(self.relu(self.bn14(x)))
        
        x = self.conv16(self.relu(self.bn15(x)))
        
        x = self.relu(self.bn16(x))
        
        x = self.avgpool(x).squeeze()
        x = self.linear1(x)
        return x