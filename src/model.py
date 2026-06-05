import torch.nn as nn
import torch
import torchvision.models as models
class VoicePathologyModel(nn.Module):
    def __init__(self):
        super(VoicePathologyModel, self).__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64,2)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)
class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
    def forward(self, x):
        return self.block(x)
class ModifiedResNet(nn.Module):
    def __init__(self):
        super(ModifiedResNet, self).__init__()
        resnet = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
        org_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(
            in_channels = 1,
            out_channels = org_conv.out_channels,
            kernel_size = org_conv.kernel_size,
            stride = org_conv.stride,
            padding = org_conv.padding,
            bias = (org_conv.bias is not None)
        )
        with torch.no_grad():
            resnet.conv1.weight.copy_(org_conv.weight.mean(dim=1, keepdim=True))
        self.feature_extractor = resnet
        self.feature_extractor.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512,2)
        )
    def forward(self, x):
        x = self.feature_extractor(x)
        return x
# class LinearBlock(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(LinearBlock, self).__init__()
#         self.block = nn.Sequential(
#             nn.Linear(in_channel,out_channel),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(out_channel,2)
#         )
#     def forward(self, x):
#         return self.block(x)
# model = VoicePathologyModel()
# dummy_x = torch.randn(1,1,128,63)
# output = model(dummy_x)
# print(output.shape)