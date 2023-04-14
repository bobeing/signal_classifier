import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()
        
        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias =False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        
        self.shortcut = nn.Sequential
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=1, stride=stride, bias = False),
                nn.BatchNorm2d(out_channels*self.expansion)
            )
            
            
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
    
class BottleNeck(nn.Module):
    """
    Residual block for resnet over 50 layers
    """
    
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )
        
        self.shorcut = nn.Sequential()
        
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shorcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride = stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )
            
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shorcut(x))
    
    
class ResNet(nn.Module):
    
    def __init__(self, block, num_block, num_classes = 10):
        super().__init__()
        
        self.in_channels = 64
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2_x = self.__make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self.__make_layer(block, 128, num_block[1], stride=2)
        self.conv4_x = self.__make_layer(block, 256, num_block[2], stride=2)
        self.conv5_x = self.__make_layer(block, 512, num_block[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        self.__init_layer()
        
    def __make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
            
        return nn.Sequential(*layers)
    
    def __init_layer(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avgpool(output)
        output = output.view(output.size(0), -1)
        # print(output.size)
        output = self.fc(output)
        # print(output.size())
        
        return output


class Model:
    def resnet18(self):
        return ResNet(BasicBlock, [2,2,2,2])

    def resnet34(self):
        return ResNet(BasicBlock, [3,4,6,3])

    def resnet50(self):
        return ResNet(BottleNeck, [3,4,6,3])

    def resnet101(self):
        return ResNet(BottleNeck, [3,4,23,3])

    def resnet152(self):
        return ResNet(BottleNeck, [3,8,36,3])